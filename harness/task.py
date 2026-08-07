"""Frontier LLM fact-checking evaluation — Inspect AI task.

The harness for running the benchmark. Framework-standard evaluation
that integrates with Inspect View for visualisation and produces
reproducible logs.

Input format — each record must have at minimum:
    claim    : str  — the claim text
    date     : str  — the epistemic anchor date (YYYY-MM-DD)
    category : str  — topic / domain (optional)
    verification_id : str  — opaque join key back to the source system (optional;
                       passed through to the result row verbatim). Not a
                       gold label — no verdict/conclusion field is read or
                       persisted anywhere in this repo.

Usage — one model at a time:
    inspect eval harness/task.py -T model_key=claude-fable-5
    inspect eval harness/task.py -T model_key=gpt-5.6-search
    inspect eval harness/task.py -T model_key=gemini-3-retrieval
    inspect eval harness/task.py -T model_key=sonar-deep-research
    inspect eval harness/task.py -T model_key=grok-4.5-search

View results:
    inspect view

Bundled runs (inspect studies/llm-disagreement/data/results.json between bundles) — one model, 50
claims at a time. Inspect's --limit takes a 1-indexed, inclusive range
(--limit 1-50 means samples 1 through 50) — all bundles write into the
same studies/llm-disagreement/data/results.jsonl:
    inspect eval harness/task.py -T model_key=claude-fable-5 --limit 1-50
    inspect eval harness/task.py -T model_key=claude-fable-5 --limit 51-100
    inspect eval harness/task.py -T model_key=claude-fable-5 --limit 101-150

Re-running a range is safe: samples that already have a successful row for
this model in out_path are skipped without an API call, and errored rows
are retried with their line replaced in place (see _load_done_ids /
_append_and_resnapshot below).

In addition to Inspect's own .eval log, every scored sample is also
upserted into studies/llm-disagreement/data/results.jsonl and the deduped
studies/llm-disagreement/data/results.json snapshot
is rewritten, so the Lenz DB import script always reads a
current, duplicate-free file. Override the path with `-T out_path=...` to
keep a debugging run out of the shared file.

claude-fable-5 fallback: any failed call (refusal, timeout, rate limit, the
ZDR-retention 400, etc.) retries once against claude-opus-4-8 before the
sample is scored as an error (see PROVIDER_CONFIG['claude-fable-5']['fallback']
in providers.py and factcheck_solver below). The result row still reports
model='claude-fable-5' (keeps the panel's 5-model-per-claim shape intact for
downstream aggregation), but carries `fallback_used: true` and
`fallback_model: 'claude-opus-4-8'` so the substitution is fully traceable —
filter on `fallback_used` before treating a row as a genuine Fable 5 answer.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import Generate, TaskState, solver

# Inspect loads this file as a top-level module (`inspect eval
# harness/task.py` from the repo root), so the sibling harness modules are
# not importable via package syntax — put this directory on sys.path and
# keep the flat imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm import build_provider  # noqa: E402
from providers import PROVIDER_CONFIG  # noqa: E402

load_dotenv()

# SDK auth-failure exceptions can echo partial key material, and this lands
# in Inspect's persisted eval log — scrub before anything is stored.
_SECRET_PATTERN = re.compile(r'\b(sk-[A-Za-z0-9_-]{10,}|xai-[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{10,}|pplx-[A-Za-z0-9_-]{10,})\b')


def _scrub_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub('[REDACTED]', text)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    'You are a fact-checker. Evaluate the truthfulness of the claim as of the stated date. '
    'Respond with JSON only.'
)

PROMPT_TEMPLATE = (
    'Evaluate this claim as of {date}:\n'
    '"{claim}"\n'
    '\n'
    'Choose exactly one verdict from the scale below:\n'
    '  "True" — the claim is accurate\n'
    '  "Mostly True" — the claim is largely accurate with minor caveats or omissions\n'
    '  "Mixed" — the claim has both accurate and inaccurate elements\n'
    '  "Mostly False" — the claim is largely inaccurate with some basis in fact\n'
    '  "False" — the claim is inaccurate\n'
    '\n'
    'Respond with a JSON object containing exactly these fields:\n'
    '  "reasoning": 2-4 sentences of claim analysis and verdict justification\n'
    '  "verdict": one of the five labels above\n'
    '  "confidence_level": your level of certainty in the verdict on a 1 to 10 integer scale '
    '(1 = completely uncertain, 10 = fully certain)'
)

VERDICT_JSON_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'reasoning': {'type': 'string', 'description': '2-4 sentence chain-of-thought analysis.'},
        'verdict': {'type': 'string', 'enum': ['True', 'Mostly True', 'Mixed', 'Mostly False', 'False']},
        'confidence_level': {'type': 'integer', 'minimum': 1, 'maximum': 10},
    },
    'required': ['reasoning', 'verdict', 'confidence_level'],
    'additionalProperties': False,
}

VERDICT_BUCKETS = ('True', 'Mostly True', 'Mixed', 'Mostly False', 'False')
_NORMALIZER = {v.lower(): v for v in VERDICT_BUCKETS}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _record_to_sample(record: dict) -> Sample:
    # No `target=` — the claims corpus has no gold verdict by design (see the
    # scorer below), so Sample's default empty target is the honest value,
    # not a lookup into a field that will never be there.
    #
    # id: prefer verification_id (guaranteed unique, short). Truncating claim text
    # to 64 chars is NOT collision-safe — near-duplicate claim variants that
    # share a long common prefix (e.g. templated A/B claim submissions)
    # truncate to the same id, and Inspect rejects the dataset outright with
    # "duplicate sample ids". Fall back to the FULL (untruncated) claim text
    # for older claims.json files that predate verification_id — full text is
    # unique across this dataset (claims were selected with pairwise
    # distance >= 0.10), truncated text is not.
    sample_id = record.get('verification_id') or record['claim']
    return Sample(
        input=PROMPT_TEMPLATE.format(
            date=record.get('date', 'unknown'),
            claim=record['claim'],
        ),
        id=sample_id,
        metadata={
            'claim': record['claim'],
            'date': record.get('date', ''),
            'category': record.get('category', ''),
            'verification_id': record.get('verification_id', ''),
        },
    )

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_decoder = json.JSONDecoder()


def _parse_verdict(raw: str) -> tuple[str, int, str]:
    """Return (verdict, confidence, reasoning) or ('', 0, '') on failure.

    Same parse contract as the retired pre-Inspect harness (see git history): same
    trailing-content tolerance (raw_decode, not loads) and the same 1-10
    confidence range check, so the Inspect harness and the harvest script
    don't grade identical model output differently.
    """
    if not raw:
        return '', 0, ''
    try:
        md = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', raw)
        text = md.group(1) if md else raw
        text = text.strip()
        for i, ch in enumerate(text):
            if ch in ('{', '['):
                try:
                    obj, _ = _decoder.raw_decode(text[i:])
                    break
                except json.JSONDecodeError:
                    continue
        else:
            return '', 0, ''
        if not isinstance(obj, dict):
            return '', 0, ''
        verdict = _NORMALIZER.get(str(obj.get('verdict', '')).strip().lower(), '')
        if not verdict:
            return '', 0, str(obj.get('reasoning', ''))
        confidence = int(obj.get('confidence_level', 0))
        if not (1 <= confidence <= 10):
            return '', 0, ''
        reasoning = str(obj.get('reasoning', ''))
        return verdict, confidence, reasoning
    except Exception:
        return '', 0, ''

# ---------------------------------------------------------------------------
# Resume — skip samples with a successful prior attempt for this model_key,
# same load-prior/done-set semantics as the retired pre-Inspect harness
# (git history). task.py previously had no resume awareness at all: re-running
# the same --limit range re-evaluated (and re-billed) every sample in it,
# including ones that already succeeded. Errored rows are deliberately NOT
# treated as done, so a resumed run retries them — most failures (timeouts,
# rate limits) are worth another try. claude-fable-5 failures specifically
# (safety-classifier refusals included) get an in-run retry against Opus 4.8
# first, via cfg['fallback'] in factcheck_solver below — this resume path is
# the second line of defense for the rarer case where both attempts fail.
# ---------------------------------------------------------------------------

def _load_done_ids(out_path: Path, model_key: str) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a prior crash
            if row.get('model') != model_key or row.get('error'):
                continue
            # Matches _record_to_sample's id: verification_id if present, else
            # the full claim text (fallback for pre-verification_id
            # claims.json rows).
            done.add(row.get('verification_id') or row.get('claim', ''))
    return done


# ---------------------------------------------------------------------------
# results.jsonl / results.json — same shape and dedup semantics as
# the retired harness, so historical and current output stay interchangeable.
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _cell_key(row: dict) -> tuple[str, str]:
    return (row.get('verification_id') or row.get('claim', ''), row.get('model', ''))


def _append_and_resnapshot(row: dict, out_path: Path) -> None:
    """Upsert `row` into out_path (jsonl, one row per (claim, model)) and
    rewrite the deduped .json snapshot alongside it.

    NOTE: despite the name (kept for continuity with the retired harness, which is
    still genuinely append-only), this rewrites the whole file rather than
    appending — a re-scored cell replaces its existing line in place instead
    of adding a second line for the same (claim, model). The retired harness kept
    true append-only semantics (crash-safe, no read-modify-write), with
    dedup only in its separate results.json snapshot; this version trades
    that crash-safety margin for never having two rows in results.jsonl
    itself, on the reasoning that this dataset is small enough (low hundreds
    of KB) that a full rewrite per cell is cheap, and the write is done to a
    temp file + atomic os.replace so a crash mid-write can't corrupt the
    existing file.

    Recomputes the snapshot from the full jsonl on every call — not
    incrementally in memory — so it's always correct even if the eval is
    interrupted mid-run; no dependency on an "eval finished" hook.

    Lock covers the whole read-modify-write: Inspect may run solver/scorer
    calls across real OS threads (blocking synchronous SDK calls can't run
    on a single asyncio event loop without serializing everything), so
    concurrent scorer invocations can race on this file exactly like
    the retired harness's ThreadPoolExecutor workers did.
    """
    with _write_lock:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows: list[dict] = []
        if out_path.exists():
            with out_path.open(encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # tolerate a truncated last line from a prior crash

        # dict preserves insertion order and updating an existing key keeps
        # its original position — so a re-scored cell rewrites its line in
        # place instead of moving to the end of the file.
        by_key: dict[tuple[str, str], dict] = {_cell_key(r): r for r in rows}
        by_key[_cell_key(row)] = row
        deduped = list(by_key.values())

        tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
        with tmp_path.open('w', encoding='utf-8') as f:
            for r in deduped:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        tmp_path.replace(out_path)

        out_path.with_suffix('.json').write_text(
            json.dumps(deduped, ensure_ascii=False, indent=2), encoding='utf-8'
        )


# ---------------------------------------------------------------------------
# Solver — uses llm.py providers directly so all provider-specific config
# (thinking, web search, reasoning effort) is preserved without remapping.
# ---------------------------------------------------------------------------

@solver
def factcheck_solver(model_key: str = 'gpt-5.6-search', done_ids: frozenset[str] = frozenset()):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Resume: this sample already has a successful row for model_key in
        # out_path. Skip the API call entirely rather than re-billing it —
        # checked here (per-sample, post-slice) rather than by filtering the
        # dataset in factcheck() below, because Inspect's --limit N-M slices
        # the dataset by POSITION after the task function returns; filtering
        # first would shift every later sample's position and silently point
        # --limit at the wrong claims.
        if str(state.sample_id) in done_ids:
            state.metadata.update({'skipped_resume': True, 'model_key': model_key})
            state.messages.append(ChatMessageAssistant(content='(skipped — already scored)'))
            state.output = ModelOutput.from_content(model=PROVIDER_CONFIG[model_key]['api_model'], content='(skipped — already scored)')
            state.completed = True
            return state

        cfg = PROVIDER_CONFIG[model_key]
        provider = build_provider(cfg)

        t0 = time.monotonic()
        error = ''
        raw = ''
        try:
            # provider.complete() is a plain blocking call (the Anthropic/
            # OpenAI/Gemini SDKs are synchronous here) — awaiting it directly
            # with no thread offload would occupy Inspect's single event-loop
            # thread for the whole request, so "concurrent" samples couldn't
            # actually make progress while one was in flight (confirmed
            # empirically: 4 fake 2s-blocking samples took ~11s, not ~2s).
            # to_thread() runs it on a worker thread instead, freeing the
            # event loop to run other samples' solve() coroutines for real.
            raw = await asyncio.to_thread(
                provider.complete,
                system=SYSTEM_PROMPT,
                user=state.user_prompt.text,
                json_schema=VERDICT_JSON_SCHEMA if provider.supports_json_schema else None,
            ) or ''
        except Exception as exc:
            error = _scrub_secrets(f'{type(exc).__name__}: {exc}')

        # Fallback: any failed call retries once against cfg['fallback'] (only
        # claude-fable-5 sets this today — Anthropic refusals/ZDR-400s/timeouts
        # all count, per-provider distinction isn't worth the complexity). The
        # row still reports model_key='claude-fable-5' (keeps the panel's
        # 5-model-per-claim join intact for downstream aggregation) but
        # cost_eur/latency_s/sources fold in the fallback attempt, and
        # fallback_used/fallback_model make the substitution fully traceable.
        primary_sources = list(provider.last_sources)
        fallback_used = False
        fallback_model = ''
        fallback_cfg = cfg.get('fallback')
        if error and fallback_cfg:
            primary_error = error
            fallback_model = fallback_cfg['api_model']
            fallback_provider = build_provider(fallback_cfg)
            try:
                raw = await asyncio.to_thread(
                    fallback_provider.complete,
                    system=SYSTEM_PROMPT,
                    user=state.user_prompt.text,
                    json_schema=VERDICT_JSON_SCHEMA if fallback_provider.supports_json_schema else None,
                ) or ''
                fallback_used = True
                primary_cost = provider.cost_eur()
                error = ''
                provider = fallback_provider
                cost_eur = primary_cost + provider.cost_eur()
            except Exception as exc:
                # Both attempts failed — keep the primary's failure reason (the
                # one downstream refusal-classification keys off, per llm.py's
                # empty-response RuntimeError comment) instead of discarding it,
                # and still bill the primary attempt's cost.
                primary_cost = provider.cost_eur()
                fallback_error = _scrub_secrets(f'{type(exc).__name__}: {exc}')
                error = f'{primary_error} | fallback {model_key}->{fallback_model} also failed: {fallback_error}'
                provider = fallback_provider
                cost_eur = primary_cost + provider.cost_eur()
        else:
            cost_eur = provider.cost_eur()

        # Merge sources gathered by both attempts — a refused/errored primary
        # call can still have run web searches before failing, and cost_eur
        # above already sums both attempts' billing, so sources should too
        # rather than silently dropping the primary's on a fallback.
        sources = primary_sources + [s for s in provider.last_sources if s not in primary_sources]

        state.metadata.update({
            'cost_eur': round(cost_eur, 6),
            'latency_s': round(time.monotonic() - t0, 2),
            'sources': sources,
            'error': error,
            'model_key': model_key,
            'fallback_used': fallback_used,
            'fallback_model': fallback_model,
        })

        content = raw or error or '(no response)'
        state.messages.append(ChatMessageAssistant(content=content))
        # Reflect the model that actually produced `content` in Inspect's own
        # transcript viewer, even though the results.jsonl row keeps
        # model_key='claude-fable-5' for the panel join (see fallback_used above).
        state.output = ModelOutput.from_content(
            model=fallback_model if fallback_used else cfg['api_model'], content=content
        )
        state.completed = True
        return state

    return solve

# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
#
# No accuracy metric: the claims corpus carries verification_id (an opaque join key
# back to the source system) but deliberately no gold verdict field.
# Scoring against target.text (always '') would let Inspect's accuracy()
# metric silently report 0% for every model on every run — a number that
# looks like a real result but measures nothing. This scorer instead just
# records the verdict/reasoning/cost as metadata, viewable via `inspect
# view`; use a gold-label join keyed on verification_id to actually grade
# harvested verdicts.

@scorer(metrics=[])
def factcheck_scorer(out_path: str = 'studies/llm-disagreement/data/results.jsonl'):
    out = Path(out_path)

    async def score(state: TaskState, target: Target) -> Score:
        if state.metadata.get('skipped_resume'):
            # Already has a successful row in out_path from a prior run —
            # don't append a second (redundant) row for the same cell.
            return Score(
                value='(skipped)',
                answer='(skipped)',
                explanation='Skipped — already scored in a prior run of this out_path.',
                metadata={'skipped_resume': True},
            )

        provider_error = state.metadata.get('error', '')

        if provider_error:
            # Mirrors the retired harness's `if error:` branch — provider call
            # itself failed, so state.output.completion holds the error
            # text (the solver's fallback content), not a model response.
            # Don't parse it as one.
            verdict, confidence, reasoning = '', 0, ''
            raw_response = ''
            row_error = provider_error
        else:
            raw_response = state.output.completion
            verdict, confidence, reasoning = _parse_verdict(raw_response)
            row_error = '' if verdict else 'parse_error'

        row = {
            'claim': state.metadata.get('claim', ''),
            'date': state.metadata.get('date', ''),
            'category': state.metadata.get('category', ''),
            'verification_id': state.metadata.get('verification_id', ''),
            'model': state.metadata.get('model_key', ''),
            'verdict': verdict,
            'reasoning': reasoning,
            'confidence': confidence,
            'cost_eur': state.metadata.get('cost_eur', 0.0),
            'latency_s': state.metadata.get('latency_s', 0.0),
            'error': row_error,
            'raw_response': raw_response,
            'sources': state.metadata.get('sources', []),
            'fallback_used': state.metadata.get('fallback_used', False),
            'fallback_model': state.metadata.get('fallback_model', ''),
        }
        # Same reasoning as the solver's to_thread wrap: this does blocking
        # file I/O (read the whole jsonl, rewrite the json snapshot) — now
        # that solver calls genuinely run concurrently, scorer calls will
        # too, so this needs to be off the event loop as well. _write_lock
        # (inside _append_and_resnapshot) still serializes the actual file
        # access across those concurrent threads.
        await asyncio.to_thread(_append_and_resnapshot, row, out)

        return Score(
            value=verdict or '(parse error)',
            answer=verdict or '(parse error)',
            explanation=(
                f'model={verdict or "none"}  '
                f'confidence={confidence}  '
                f'cost=€{state.metadata.get("cost_eur", 0):.4f}  '
                f'{state.metadata.get("latency_s", 0):.1f}s\n'
                f'{reasoning}'
            ),
            metadata={
                'verdict': verdict,
                'confidence': confidence,
                'cost_eur': state.metadata.get('cost_eur', 0.0),
                'latency_s': state.metadata.get('latency_s', 0.0),
                'sources': state.metadata.get('sources', []),
                'error': row_error,
                'reasoning': reasoning,
                'model_key': state.metadata.get('model_key', ''),
                'category': state.metadata.get('category', ''),
            },
        )

    return score

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@task
def factcheck(
    model_key: str = 'gpt-5.6-search',
    claims_file: str = 'studies/llm-disagreement/data/claims.json',
    out_path: str = 'studies/llm-disagreement/data/results.jsonl',
):
    dataset = json_dataset(claims_file, _record_to_sample)

    # Deliberately NOT filtered out of `dataset` here — Inspect's --limit N-M
    # slices dataset[N-1:M] by POSITION after this function returns (see
    # inspect_ai._eval.task.util.sample_slice), so removing already-done
    # samples first would shift every later sample's position and silently
    # point --limit at the wrong claims. Skipping instead happens per-sample
    # in factcheck_solver, which runs after the positional slice.
    done_ids = frozenset(_load_done_ids(Path(out_path), model_key))
    if done_ids:
        print(f'Resume: {len(done_ids)} sample(s) already succeeded for {model_key} in {out_path} — will skip, not re-bill.\n')

    return Task(
        dataset=dataset,
        solver=factcheck_solver(model_key=model_key, done_ids=done_ids),
        scorer=factcheck_scorer(out_path=out_path),
    )
