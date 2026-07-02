"""Frontier LLM fact-checking evaluation — Inspect AI task.

Framework-standard evaluation that integrates with Inspect View
for visualisation and produces reproducible logs.

Input format — each record must have at minimum:
    claim    : str  — the claim text
    date     : str  — the epistemic anchor date (YYYY-MM-DD)
    category : str  — topic / domain (optional)

Usage — one model at a time:
    inspect eval task.py -T model_key=claude-fable-5
    inspect eval task.py -T model_key=gpt-5.5-search
    inspect eval task.py -T model_key=gemini-3-retrieval
    inspect eval task.py -T model_key=sonar-deep-research
    inspect eval task.py -T model_key=grok-4.3-search

View results:
    inspect view
"""

from __future__ import annotations

import json
import re
import time
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
    return Sample(
        input=PROMPT_TEMPLATE.format(
            date=record.get('date', 'unknown'),
            claim=record['claim'],
        ),
        id=str(record.get('claim', ''))[:64],
        metadata={
            'claim': record['claim'],
            'date': record.get('date', ''),
            'category': record.get('category', ''),
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
            raw = provider.complete(
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
# No accuracy metric: data/claims.json is deliberately built without a gold
# verdict field (this repo stands alone, with no Lenz-internal identifiers —
# see README). Scoring against target.text (always '') would let Inspect's
# accuracy() metric silently report 0% for every model on every run — a
# number that looks like a real result but measures nothing. This scorer
# instead just records the verdict/reasoning/cost as metadata, viewable via
# `inspect view`; use compare.py (or your own gold-label join) to actually
# grade harvested verdicts.

@scorer(metrics=[])
def factcheck_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        verdict, confidence, reasoning = _parse_verdict(state.output.completion)

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
                'error': state.metadata.get('error', ''),
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
):
    return Task(
        dataset=json_dataset(claims_file, _record_to_sample),
        solver=factcheck_solver(model_key=model_key),
        scorer=factcheck_scorer(),
    )
