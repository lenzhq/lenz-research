"""Frontier LLM fact-checking benchmark.

BACKUP path — task.py (Inspect AI) is the primary harness for now. This
script has no framework dependency (plain stdlib + vendor SDKs), so it
stays here as a fallback if Inspect itself is ever the blocker. Both write
the same data/results.jsonl / data/results.json shape and are safe to
interleave.

Reads a JSON/JSONL file of claims, evaluates each claim × model in parallel,
and writes results to data/results.jsonl (append-only streaming log) and
data/results.json (deduped final snapshot, one row per (claim, model)).

Input format — each record must have at minimum:
    claim    : str   — the claim text
    date     : str   — the epistemic anchor date (YYYY-MM-DD)
    category : str   — topic / domain (optional)

Any extra fields (e.g. verdict, share_id) are ignored.

Resumable: re-running the same --out path skips (claim, model) cells that
already succeeded and retries ones that previously errored (most failures
are transient — timeout, rate limit). Ctrl+C cancels every not-yet-started
cell instead of burning the full remaining budget; in-flight API calls
already running will still complete and bill.

Usage:
    python harvest.py --claims data/claims.json
    python harvest.py --claims data/claims.json --models gpt-5.5-search,claude-fable-5

Bundled runs (inspect data/results.json between bundles) — one model, 50
claims at a time, all writing into the same --out file:
    python harvest.py --claims data/claims.json --models claude-fable-5 --offset 0   --limit 50
    python harvest.py --claims data/claims.json --models claude-fable-5 --offset 50  --limit 50
    python harvest.py --claims data/claims.json --models claude-fable-5 --offset 100 --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from llm import build_provider
from providers import PROVIDER_CONFIG

load_dotenv()
logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

# Scrub API-key-shaped substrings from persisted error text — SDK exceptions
# (auth failures especially) can echo partial key material, and results.jsonl
# is a committed reproducibility artifact, not a private log.
_SECRET_PATTERN = re.compile(r'\b(sk-[A-Za-z0-9_-]{10,}|xai-[A-Za-z0-9_-]{10,}|AIza[A-Za-z0-9_-]{10,}|pplx-[A-Za-z0-9_-]{10,})\b')
# Model-generated text (reasoning, claim echoes) is untrusted output printed
# straight to the terminal — strip control/escape chars so it can't spoof
# output, retitle the terminal, or write to the clipboard via OSC 52.
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')


def _scrub_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub('[REDACTED]', text)


def _sanitize_for_terminal(text: str) -> str:
    return _CONTROL_CHARS.sub('', text)

# ---------------------------------------------------------------------------
# Prompt
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
    '  "True"         — the claim is accurate\n'
    '  "Mostly True"  — the claim is largely accurate with minor caveats or omissions\n'
    '  "Mixed"        — the claim has both accurate and inaccurate elements\n'
    '  "Mostly False" — the claim is largely inaccurate with some basis in fact\n'
    '  "False"        — the claim is inaccurate\n'
    '\n'
    'Respond with a JSON object containing exactly these fields:\n'
    '  "reasoning": 2-4 sentences of claim analysis and verdict justification\n'
    '  "verdict":   one of the five labels above\n'
    '  "confidence_level": your level of certainty in the verdict on a 1 to 10 integer scale '
    '(1 = completely uncertain, 10 = fully certain).'
)

VERDICT_JSON_SCHEMA: dict = {
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
WORKERS = 5

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_VERDICT_NORMALIZER = {v.lower(): v for v in VERDICT_BUCKETS}


_decoder = json.JSONDecoder()


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction — handles markdown fences, <think> prefixes,
    and trailing prose after the JSON object (e.g. Perplexity citation footnotes).

    Uses raw_decode instead of loads: loads requires the ENTIRE remainder of
    the string to be valid JSON, so "{...}\\n\\nSources: [1] ..." fails every
    scan position even though a valid object is right there.
    """
    md_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', text)
    if md_match:
        text = md_match.group(1)
    else:
        fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*)', text)
        if fence_match:
            text = fence_match.group(1)
    text = text.strip()
    for i, ch in enumerate(text):
        if ch in ('{', '['):
            try:
                obj, _ = _decoder.raw_decode(text[i:])
                return obj
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def parse_response(raw: str | None) -> tuple[str, int, str, str]:
    """Parse model response into (verdict, confidence, reasoning, status).

    status is 'parsed' on success or 'error' on any failure.
    """
    if not raw:
        return ('', 0, '', 'error')
    try:
        result = _extract_json(raw)
    except Exception:
        return ('', 0, '', 'error')
    if not isinstance(result, dict):
        return ('', 0, '', 'error')
    verdict = _VERDICT_NORMALIZER.get(str(result.get('verdict', '')).strip().lower())
    if not verdict:
        return ('', 0, str(result.get('reasoning', '')), 'error')
    try:
        confidence = int(result.get('confidence_level', 0))
    except (TypeError, ValueError):
        return ('', 0, '', 'error')
    if not (1 <= confidence <= 10):
        return ('', 0, '', 'error')
    return (verdict, confidence, str(result.get('reasoning', '')), 'parsed')


def _load_prior_results(out_path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    """Read out_path's existing rows (if any), for resuming an interrupted run.

    Returns (all_prior_rows, done_pairs). done_pairs are (claim_text, model)
    pairs with a SUCCESSFUL prior attempt — safe to skip on resume. Errored
    rows are kept in all_prior_rows (so the final snapshot stays complete)
    but are deliberately NOT in done_pairs, so a resume retries them — most
    harvest failures are transient (timeout, rate limit) rather than
    permanent, and re-persisting the same failure forever helps no one.
    """
    if not out_path.exists():
        return [], set()
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    with out_path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a truncated last line from a prior crash
            rows.append(row)
            if not row.get('error'):
                done.add((row.get('claim', ''), row.get('model', '')))
    return rows, done


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(claim: dict, model: str) -> dict:
    """Evaluate one (claim × model) cell. Returns a result dict."""
    cfg = PROVIDER_CONFIG[model]
    provider = build_provider(cfg)

    user_prompt = PROMPT_TEMPLATE.format(
        date=claim.get('date', 'unknown'),
        claim=claim['claim'],
    )

    t0 = time.monotonic()
    error = ''
    raw = ''
    try:
        raw = provider.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            json_schema=VERDICT_JSON_SCHEMA if provider.supports_json_schema else None,
        ) or ''
    except Exception as exc:
        error = _scrub_secrets(f'{type(exc).__name__}: {exc}')

    latency_s = round(time.monotonic() - t0, 2)
    cost = round(provider.cost_eur(), 6)
    sources = list(provider.last_sources)

    if error:
        return {
            'claim': claim['claim'],
            'date': claim.get('date', ''),
            'category': claim.get('category', ''),
            'model': model,
            'verdict': '',
            'reasoning': '',
            'confidence': 0,
            'cost_eur': cost,
            'latency_s': latency_s,
            'error': error,
            'raw_response': '',
            'sources': sources,
        }

    verdict, confidence, reasoning, status = parse_response(raw)

    return {
        'claim': claim['claim'],
        'date': claim.get('date', ''),
        'category': claim.get('category', ''),
        'model': model,
        'verdict': verdict,
        'reasoning': reasoning,
        'confidence': confidence,
        'cost_eur': cost,
        'latency_s': latency_s,
        'error': '' if status == 'parsed' else 'parse_error',
        'raw_response': raw,
        'sources': sources,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Frontier LLM fact-check benchmark')
    parser.add_argument('--claims', required=True, help='Path to claims JSON or JSONL file')
    parser.add_argument('--out', default='data/results.jsonl', help='Streaming JSONL output (default: data/results.jsonl)')
    parser.add_argument('--models', default='', help='Comma-separated model subset (default: all)')
    parser.add_argument('--offset', type=int, default=0, help='Skip the first N claims (default: 0)')
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Take at most N claims after --offset (default: all remaining). '
             'Combine with --offset to run in bundles, e.g. --offset 0 --limit 50, '
             'then --offset 50 --limit 50 — both write into the same --out file, '
             'so results.json accumulates across bundles.',
    )
    args = parser.parse_args()

    claims_path = Path(args.claims)
    if not claims_path.exists():
        print(f'Error: {claims_path} not found', file=sys.stderr)
        sys.exit(1)

    raw_text = claims_path.read_text(encoding='utf-8').strip()
    if claims_path.suffix == '.json' or raw_text.startswith('['):
        claims = json.loads(raw_text)
    else:
        claims = [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if args.offset < 0:
        print(f'Error: --offset must be >= 0 (got {args.offset})', file=sys.stderr)
        sys.exit(1)
    if args.limit is not None and args.limit < 1:
        print(f'Error: --limit must be >= 1 (got {args.limit})', file=sys.stderr)
        sys.exit(1)
    if args.offset >= len(claims) and len(claims) > 0:
        print(f'Error: --offset {args.offset} is beyond the claims file ({len(claims)} claims total)', file=sys.stderr)
        sys.exit(1)

    n_claims_total = len(claims)
    end = args.offset + args.limit if args.limit is not None else None
    claims = claims[args.offset:end]
    if args.offset or args.limit is not None:
        print(
            f'Bundle: claims[{args.offset}:{end if end is not None else n_claims_total}] '
            f'— {len(claims)} of {n_claims_total} total claims\n'
        )

    models = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else list(PROVIDER_CONFIG)
    unknown = [m for m in models if m not in PROVIDER_CONFIG]
    if unknown:
        print(f'Unknown models: {unknown}. Available: {list(PROVIDER_CONFIG)}', file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prior_results, completed = _load_prior_results(out_path)
    cells = [(claim, model) for claim in claims for model in models]
    if completed:
        before = len(cells)
        cells = [(claim, model) for claim, model in cells if (claim['claim'], model) not in completed]
        n_prior_errors = sum(1 for r in prior_results if r.get('error'))
        print(
            f'Resuming {out_path}: {len(completed)} cells already succeeded, '
            f'{before - len(cells)} skipped, {len(cells)} remaining '
            f'({n_prior_errors} prior errors will be retried).\n'
        )

    n_total = len(cells)
    if n_total == 0:
        print('Nothing to do — every cell already succeeded in a prior run.')
        return
    print(f'Running {n_total} cells ({len(claims)} claim(s) × {len(models)} model(s))...\n')

    # Keyed by (claim_text, model) so a retried-and-now-succeeding cell
    # overwrites its old errored row instead of duplicating it in the final
    # snapshot (the streaming .jsonl file itself may still carry both lines —
    # that's an append-only log, not the deduped snapshot).
    results_by_cell: dict[tuple[str, str], dict] = {
        (r.get('claim', ''), r.get('model', '')): r for r in prior_results
    }

    n_done = 0
    n_error = 0
    interrupted = False
    file_mode = 'a' if prior_results else 'w'

    with out_path.open(file_mode, encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(run_cell, claim, model): (claim, model) for claim, model in cells}
            try:
                for fut in as_completed(futures):
                    claim, model = futures[fut]
                    n_done += 1
                    try:
                        result = fut.result()
                    except Exception as exc:
                        n_error += 1
                        print(f'  [{n_done}/{n_total}] {model:30s}  EXCEPTION: {_scrub_secrets(str(exc))}')
                        continue

                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                    results_by_cell[(result['claim'], result['model'])] = result

                    if result['error']:
                        n_error += 1
                        print(f'  [{n_done}/{n_total}] {model:30s}  ERROR  ({result["error"]})')
                    else:
                        print(
                            f'  [{n_done}/{n_total}] {model:30s}  {result["verdict"]:12s} '
                            f'(conf {result["confidence"]:>2}/10)  '
                            f'€{result["cost_eur"]:.4f}  {result["latency_s"]:.1f}s'
                        )
                        if result['reasoning']:
                            print(f'           {_sanitize_for_terminal(result["reasoning"])}')
                        print()
            except KeyboardInterrupt:
                interrupted = True
                n_queued = n_total - n_done
                print(
                    f'\nInterrupted at {n_done}/{n_total} cells. Cancelling {n_queued} not-yet-started '
                    f'cells — in-flight API calls already running will still complete and bill, '
                    f'but the rest of the queue will NOT be spent.'
                )
                # Futures already executing can't be aborted (no thread-abort
                # hook in concurrent.futures) — but everything still queued
                # is freed instead of silently running to completion.
                ex.shutdown(wait=False, cancel_futures=True)

    # Final snapshot: always written, even on interrupt, so a partial run's
    # results are immediately usable and results.json never goes stale
    # relative to results.jsonl (see compare.py's staleness warning for the
    # remaining hard-kill case this can't cover).
    all_results = list(results_by_cell.values())
    total_cost = sum(r.get('cost_eur', 0.0) for r in all_results)
    json_path = out_path.with_suffix('.json')
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding='utf-8')

    status = 'INTERRUPTED' if interrupted else 'Done'
    print(
        f'\n{status}: this run attempted {n_done}/{n_total} cells, {n_done - n_error} succeeded, '
        f'{n_error} errored. {len(all_results)} total rows, total spend so far €{total_cost:.4f}'
    )
    print(f'Streaming log : {out_path}')
    print(f'JSON results  : {json_path}')
    if interrupted:
        print('Re-run the same command to resume — already-succeeded cells are skipped automatically.')
        sys.exit(130)  # standard exit code for SIGINT


if __name__ == '__main__':
    main()
