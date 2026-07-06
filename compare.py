"""Compare model verdicts per claim.

Reads data/results.json and groups results by claim text.
Since claims are unique (pairwise distance >= 0.10), claim text is the key.

Outputs:
  stdout — summary table + disagreement breakdown

Usage:
    python compare.py
    python compare.py --results data/results.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Claim text and error strings are model-controlled/model-adjacent text
# printed straight to the terminal — strip control/escape chars so they
# can't spoof output or retitle the terminal (same guard as task.py).
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b-\x1f\x7f]')


def _sanitize_for_terminal(text: str) -> str:
    return _CONTROL_CHARS.sub('', text)


def majority_verdict(verdicts: list[str]) -> tuple[str, bool]:
    """Most common verdict among non-empty verdicts.

    Returns (verdict, is_tie). verdict is '' if there are no non-empty
    verdicts, or if the top count is tied. Counter.most_common() breaks
    ties by insertion order (i.e. the order models were iterated), which
    would otherwise make "consensus" an artifact of model naming rather
    than a real majority — callers must not silently accept a tie as a win.
    """
    counts = Counter(v for v in verdicts if v)
    if not counts:
        return '', False
    ranked = counts.most_common()
    top_count = ranked[0][1]
    is_tie = len(ranked) > 1 and ranked[1][1] == top_count
    return ('' if is_tie else ranked[0][0]), is_tie


def verdict_spread(verdicts: list[str]) -> int:
    """Number of distinct non-empty verdicts (errored/empty rows excluded)."""
    return len({v for v in verdicts if v})


def main():
    parser = argparse.ArgumentParser(description='Compare model verdicts per claim')
    parser.add_argument('--results', default='data/results.json')
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f'Error: {results_path} not found. Run task.py first (see README).')
        return

    # task.py rewrites results.json from results.jsonl after every scored
    # sample — but a hard kill (OOM, kill -9) can still land between the two
    # writes, leaving results.json stale while the jsonl has newer rows.
    jsonl_path = results_path.with_suffix('.jsonl')
    if jsonl_path.exists() and jsonl_path.stat().st_mtime > results_path.stat().st_mtime:
        print(
            f'WARNING: {jsonl_path} is newer than {results_path} — the eval may not have '
            f'finished writing the final snapshot. These statistics may be stale or incomplete.\n'
        )

    with open(results_path, encoding='utf-8') as f:
        results = json.load(f)

    if not results:
        print(f'No results in {results_path} — nothing to compare.')
        return

    # Group by claim text. Safe only because claims were selected with
    # pairwise distance >= 0.10 (see extract_diverse_claims.py) — if that
    # invariant is ever violated, or results.jsonl is accidentally
    # concatenated across two different claim sets, a collision here would
    # silently overwrite one claim's row instead of erroring.
    by_claim: dict[str, dict] = {}
    for r in results:
        claim_text = r['claim']
        existing = by_claim.get(claim_text)
        if existing is not None and existing['date'] != r.get('date', '') and existing['category'] != r.get('category', ''):
            raise ValueError(
                f'Duplicate claim text with different date/category metadata: {claim_text[:80]!r} — '
                'the claim-text join key is unsafe for this results file.'
            )
        if claim_text not in by_claim:
            by_claim[claim_text] = {
                'claim': claim_text,
                'date': r.get('date', ''),
                'category': r.get('category', ''),
                'verification_id': r.get('verification_id', ''),
                'models': {},
            }
        by_claim[claim_text]['models'][r['model']] = {
            'verdict': r.get('verdict', ''),
            'confidence': r.get('confidence', 0),
            'reasoning': r.get('reasoning', ''),
            'sources': r.get('sources', []),
            'error': r.get('error', ''),
            'cost_eur': r.get('cost_eur', 0.0),
            'latency_s': r.get('latency_s', 0.0),
        }

    # Build comparison entries
    all_models = sorted({r['model'] for r in results})
    comparison = []

    n_unanimous = 0
    n_split = 0
    n_all_error = 0
    verdict_disagreements: Counter = Counter()

    for claim_text, entry in by_claim.items():
        models = entry['models']
        verdicts = [models[m]['verdict'] for m in all_models if m in models]
        non_empty = [v for v in verdicts if v]

        consensus, tied = majority_verdict(verdicts)
        spread = verdict_spread(verdicts)
        all_error = len(non_empty) == 0
        # Unanimous among RESPONDING models — a model that errored out isn't
        # a dissenter. Previously this required every model (including
        # errored ones) to share a verdict, so a 4-agree/1-error claim fell
        # through to the split bucket as a nonsensical "1-way split".
        unanimous = (not all_error) and not tied and spread == 1

        disagreeing = [] if (tied or all_error) else [
            m for m in all_models
            if m in models and models[m]['verdict'] and models[m]['verdict'] != consensus
        ]

        entry['consensus'] = consensus
        entry['unanimous'] = unanimous
        entry['tied'] = tied
        entry['all_error'] = all_error
        entry['spread'] = spread
        entry['disagreeing_models'] = disagreeing

        comparison.append(entry)

        if all_error:
            n_all_error += 1
        elif unanimous:
            n_unanimous += 1
        else:
            n_split += 1
            label = 'tie' if tied else f'{spread}-way split'
            verdict_disagreements[label] += 1

    # Sort: most disagreement first
    comparison.sort(key=lambda e: (-e['spread'], e['category'], e['claim']))

    # Print summary
    n_claims = len(comparison)
    n_rows = len(results)
    print(f'\n{"="*60}')
    print(f'  FRONTIER MODEL COMPARISON')
    print(f'{"="*60}')
    print(f'  Claims:          {n_claims}')
    print(f'  Models:          {len(all_models)}')
    print(f'  Result rows:     {n_rows}')
    print(f'  Unanimous:       {n_unanimous}  ({100*n_unanimous/n_claims:.1f}%)')
    print(f'  Split:           {n_split}  ({100*n_split/n_claims:.1f}%)')
    print(f'  All error:       {n_all_error}')
    print()

    for label, count in sorted(verdict_disagreements.items()):
        print(f'    {label}: {count}')

    # Print top disagreements — real splits and ties, not all-error claims
    # (all_error is tracked explicitly now; consensus=='' no longer implies
    # all_error, since a genuine tie also has an empty consensus).
    splits = [e for e in comparison if not e['unanimous'] and not e['all_error']]
    if splits:
        print(f'\n--- Top {min(10, len(splits))} disagreements ---\n')
        for entry in splits[:10]:
            models = entry['models']
            tie_marker = '[TIE] ' if entry['tied'] else ''
            print(f'  [{entry["category"]}] {tie_marker}{_sanitize_for_terminal(entry["claim"][:90])}')
            for m in all_models:
                if m not in models:
                    continue
                v = models[m]['verdict'] or f'ERROR({_sanitize_for_terminal(models[m]["error"][:30])})'
                conf = models[m]['confidence']
                flag = ' <--' if m in entry['disagreeing_models'] else ''
                print(f'    {m:30s}  {v:12s}  conf={conf}{flag}')
            print()


if __name__ == '__main__':
    main()
