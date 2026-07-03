"""Frontier LLM fact-checking evaluation — Inspect AI task.

Primary harness for running the benchmark. Framework-standard evaluation
that integrates with Inspect View for visualisation and produces
reproducible logs. (harvest.py is kept as a backup path — plain-script,
no framework dependency — in case Inspect itself is ever the blocker.)

Input format — each record must have at minimum:
    claim    : str  — the claim text
    date     : str  — the epistemic anchor date (YYYY-MM-DD)
    category : str  — topic / domain (optional)
    share_id : str  — opaque join key back to the source system (optional;
                       passed through to the result row verbatim). Not a
                       gold label — no verdict/conclusion field is read or
                       persisted anywhere in this repo.

Usage — one model at a time:
    inspect eval task.py -T model_key=claude-fable-5
    inspect eval task.py -T model_key=gpt-5.5-search
    inspect eval task.py -T model_key=gemini-3-retrieval
    inspect eval task.py -T model_key=sonar-deep-research
    inspect eval task.py -T model_key=grok-4.3-search

View results:
    inspect view

Bundled runs (inspect data/results.json between bundles) — one model, 50
claims at a time. Inspect's --limit takes a 1-indexed, inclusive range
(--limit 1-50 means samples 1 through 50, NOT 0-indexed like harvest.py's
--offset/--limit) — all bundles write into the same data/results.jsonl:
    inspect eval task.py -T model_key=claude-fable-5 --limit 1-50
    inspect eval task.py -T model_key=claude-fable-5 --limit 51-100
    inspect eval task.py -T model_key=claude-fable-5 --limit 101-150

In addition to Inspect's own .eval log, every scored sample is also
appended to data/results.jsonl and the deduped data/results.json snapshot
is rewritten — byte-identical row shape to harvest.py's output, same
default path, so compare.py and the Lenz DB import script read either
harness's output interchangeably, and a later `harvest.py` resume run will
correctly skip (claim, model) pairs already scored here. Override the path
with `-T out_path=...` to keep a debugging run out of the shared file.
"""

from __future__ import annotations

import asyncio
import json
import re
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

from llm import build_provider
from providers import PROVIDER_CONFIG

load_dotenv()

# Same scrub as harvest.py: SDK auth-failure exceptions can echo partial key
# material, and this lands in Inspect's persisted eval log.
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
    # No `target=` — data/claims.json has no gold verdict by design (see the
    # scorer below), so Sample's default empty target is the honest value,
    # not a lookup into a field that will never be there.
    #
    # id: prefer share_id (guaranteed unique, short). Truncating claim text
    # to 64 chars is NOT collision-safe — near-duplicate claim variants that
    # share a long common prefix (e.g. templated A/B claim submissions)
    # truncate to the same id, and Inspect rejects the dataset outright with
    # "duplicate sample ids". Fall back to the FULL (untruncated) claim text
    # for older claims.json files that predate share_id — full text is
    # unique across this dataset (claims were selected with pairwise
    # distance >= 0.10), truncated text is not.
    sample_id = record.get('share_id') or record['claim']
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
            'share_id': record.get('share_id', ''),
        },
    )

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_decoder = json.JSONDecoder()


def _parse_verdict(raw: str) -> tuple[str, int, str]:
    """Return (verdict, confidence, reasoning) or ('', 0, '') on failure.

    Kept in parity with harvest.py's parse_response/_extract_json: same
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
# results.jsonl / results.json — same shape and dedup semantics as
# harvest.py, so the two harnesses' output is interchangeable.
# ---------------------------------------------------------------------------

_write_lock = threading.Lock()


def _append_and_resnapshot(row: dict, out_path: Path) -> None:
    """Append `row` to out_path (jsonl) and rewrite the deduped .json
    snapshot alongside it.

    Recomputes the snapshot from the full jsonl on every call — not
    incrementally in memory — so it's always correct even if the eval is
    interrupted mid-run; no dependency on an "eval finished" hook. Cheap at
    this dataset size (low hundreds of KB), same tradeoff harvest.py makes.

    Lock covers the whole append+resnapshot: Inspect may run solver/scorer
    calls across real OS threads (blocking synchronous SDK calls can't run
    on a single asyncio event loop without serializing everything), so
    concurrent scorer invocations can race on this file exactly like
    harvest.py's ThreadPoolExecutor workers do.
    """
    with _write_lock:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

        rows: list[dict] = []
        with out_path.open(encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # tolerate a truncated last line from a prior crash

        # Last write wins per (claim, model) — matches harvest.py's
        # results_by_cell dedup, so a re-scored cell doesn't duplicate.
        deduped = list({(r.get('claim', ''), r.get('model', '')): r for r in rows}.values())
        out_path.with_suffix('.json').write_text(
            json.dumps(deduped, ensure_ascii=False, indent=2), encoding='utf-8'
        )


# ---------------------------------------------------------------------------
# Solver — uses llm.py providers directly so all provider-specific config
# (thinking, web search, reasoning effort) is preserved without remapping.
# ---------------------------------------------------------------------------

@solver
def factcheck_solver(model_key: str = 'gpt-5.5-search'):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
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

        state.metadata.update({
            'cost_eur': round(provider.cost_eur(), 6),
            'latency_s': round(time.monotonic() - t0, 2),
            'sources': list(provider.last_sources),
            'error': error,
            'model_key': model_key,
        })

        content = raw or error or '(no response)'
        state.messages.append(ChatMessageAssistant(content=content))
        state.output = ModelOutput.from_content(model=cfg['api_model'], content=content)
        state.completed = True
        return state

    return solve

# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
#
# No accuracy metric: data/claims.json carries share_id (an opaque join key
# back to the source system) but deliberately no gold verdict field.
# Scoring against target.text (always '') would let Inspect's accuracy()
# metric silently report 0% for every model on every run — a number that
# looks like a real result but measures nothing. This scorer instead just
# records the verdict/reasoning/cost as metadata, viewable via `inspect
# view`; use compare.py (or a gold-label join keyed on share_id) to
# actually grade harvested verdicts.

@scorer(metrics=[])
def factcheck_scorer(out_path: str = 'data/results.jsonl'):
    out = Path(out_path)

    async def score(state: TaskState, target: Target) -> Score:
        provider_error = state.metadata.get('error', '')

        if provider_error:
            # Mirrors harvest.py's `if error:` branch — provider call
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
            'share_id': state.metadata.get('share_id', ''),
            'model': state.metadata.get('model_key', ''),
            'verdict': verdict,
            'reasoning': reasoning,
            'confidence': confidence,
            'cost_eur': state.metadata.get('cost_eur', 0.0),
            'latency_s': state.metadata.get('latency_s', 0.0),
            'error': row_error,
            'raw_response': raw_response,
            'sources': state.metadata.get('sources', []),
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
    model_key: str = 'gpt-5.5-search',
    claims_file: str = 'data/claims.json',
    out_path: str = 'data/results.jsonl',
):
    return Task(
        dataset=json_dataset(claims_file, _record_to_sample),
        solver=factcheck_solver(model_key=model_key),
        scorer=factcheck_scorer(out_path=out_path),
    )
