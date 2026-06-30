"""Reproduce the frontier LLM fact-check benchmark.

Reads a JSONL file of claims, evaluates each claim × model in parallel,
and writes per-(claim × model) results to an output JSONL file.

Usage:
    python harvest.py --claims data/claims.jsonl --out data/my_results.jsonl

Compare your output against data/results.jsonl to verify reproducibility.
Results may differ slightly due to model non-determinism, but agreement
rates should be within a few percentage points.
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

from llm import build_provider
from providers import PROVIDER_CONFIG

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')

# ---------------------------------------------------------------------------
# Prompt (v4 — matches the published benchmark)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    'You are a rigorous fact-checker. Evaluate claims based on the best available '
    'evidence as of the stated date. Respond with JSON only — no prose outside the JSON.'
)

PROMPT_TEMPLATE = (
    'Evaluate this claim as of {date}:\n'
    '"{claim}"\n'
    '\n'
    'Choose exactly one verdict from the scale below:\n'
    '  "True"         — the claim is accurate and well-supported by evidence\n'
    '  "Mostly True"  — the claim is largely accurate with minor caveats or omissions\n'
    '  "Mixed"        — the claim has both accurate and inaccurate elements\n'
    '  "Mostly False" — the claim is largely inaccurate with some basis in fact\n'
    '  "False"        — the claim is inaccurate or unsupported by evidence\n'
    '\n'
    'Respond with a JSON object containing exactly these fields:\n'
    '  "reasoning": 2-4 sentences of evidence-based analysis\n'
    '  "verdict":   one of the five labels above\n'
    '  "confidence_level": an integer from 1 to 10 (1 = completely uncertain, 10 = fully certain)\n'
    '\n'
    'Use a low confidence_level (1-3) only when the claim is genuinely contested among '
    'experts or you lack sufficient evidence for a reliable verdict.\n'
    '\n'
    'Example output:\n'
    '{{"reasoning": "...", "verdict": "Mostly True", "confidence_level": 7}}'
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


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction — handles markdown fences and <think> prefixes."""
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
                return json.loads(text[i:])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def parse_response(raw: str | None) -> tuple[str, int, str, str]:
    """Parse a model response into (verdict, confidence, reasoning, status).

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


# ---------------------------------------------------------------------------
# Cell runner
# ---------------------------------------------------------------------------

def run_cell(claim: dict, model: str) -> dict:
    """Evaluate one (claim × model) cell. Returns a result dict."""
    cfg = PROVIDER_CONFIG[model]
    provider = build_provider(cfg)

    user_prompt = PROMPT_TEMPLATE.format(
        date=claim.get('submission_date', 'unknown'),
        claim=claim['atomic_claim'],
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
        error = f'{type(exc).__name__}: {exc}'

    latency_s = round(time.monotonic() - t0, 2)
    cost = round(provider.cost_eur(), 6)

    if error:
        return {
            'atomic_claim': claim['atomic_claim'],
            'domain': claim.get('domain', ''),
            'submission_date': claim.get('submission_date', ''),
            'lenz_verdict': claim.get('conclusion_label', ''),
            'model': model,
            'verdict': '',
            'reasoning': '',
            'confidence': 0,
            'agrees_with_lenz': None,
            'cost_eur': cost,
            'latency_s': latency_s,
            'error': error,
            'raw_response': '',
        }

    verdict, confidence, reasoning, status = parse_response(raw)
    lenz_verdict = claim.get('conclusion_label', '')
    agrees = (verdict == lenz_verdict) if (status == 'parsed' and lenz_verdict) else None

    return {
        'atomic_claim': claim['atomic_claim'],
        'domain': claim.get('domain', ''),
        'submission_date': claim.get('submission_date', ''),
        'lenz_verdict': lenz_verdict,
        'model': model,
        'verdict': verdict,
        'reasoning': reasoning,
        'confidence': confidence,
        'agrees_with_lenz': agrees,
        'cost_eur': cost,
        'latency_s': latency_s,
        'error': '' if status == 'parsed' else 'parse_error',
        'raw_response': raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Frontier LLM fact-check benchmark')
    parser.add_argument('--claims', required=True, help='Path to claims JSONL file')
    parser.add_argument('--out', required=True, help='Path to write results JSONL')
    parser.add_argument('--models', default='', help='Comma-separated model subset (default: all 5)')
    args = parser.parse_args()

    claims_path = Path(args.claims)
    if not claims_path.exists():
        print(f'Error: {claims_path} not found', file=sys.stderr)
        sys.exit(1)

    claims = [json.loads(line) for line in claims_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    models = [m.strip() for m in args.models.split(',') if m.strip()] if args.models else list(PROVIDER_CONFIG)

    unknown = [m for m in models if m not in PROVIDER_CONFIG]
    if unknown:
        print(f'Unknown models: {unknown}. Available: {list(PROVIDER_CONFIG)}', file=sys.stderr)
        sys.exit(1)

    cells = [(claim, model) for claim in claims for model in models]
    n_total = len(cells)
    print(f'Running {n_total} cells ({len(claims)} claim(s) × {len(models)} model(s))...\n')

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_error = 0
    total_cost = 0.0

    with out_path.open('w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(run_cell, claim, model): (claim, model) for claim, model in cells}
            for fut in as_completed(futures):
                claim, model = futures[fut]
                n_done += 1
                try:
                    result = fut.result()
                except Exception as exc:
                    n_error += 1
                    print(f'  [{n_done}/{n_total}] {model:30s}  EXCEPTION: {exc}')
                    continue

                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                total_cost += result['cost_eur']

                if result['error']:
                    n_error += 1
                    print(f'  [{n_done}/{n_total}] {model:30s}  ERROR  ({result["error"]})')
                else:
                    match = '✓' if result['agrees_with_lenz'] else ('?' if result['agrees_with_lenz'] is None else '✗')
                    print(
                        f'  [{n_done}/{n_total}] {model:30s}  {result["verdict"]:12s} '
                        f'(conf {result["confidence"]:>2}/10)  lenz={result["lenz_verdict"]:12s} {match}  '
                        f'€{result["cost_eur"]:.4f}  {result["latency_s"]:.1f}s'
                    )
                    if result['reasoning']:
                        print(f'           {result["reasoning"]}')
                    print()

    print(f'\nDone: {n_done - n_error}/{n_done} succeeded  errors={n_error}  total_cost=€{total_cost:.4f}')
    print(f'Results written to {out_path}')


if __name__ == '__main__':
    main()
