"""Frontier-LLM provider configuration for the fact-checking benchmark.

Five models evaluated at a medium reasoning tier — mid-level thinking + live
retrieval + native JSON schema enforcement:

  claude-fable-5      Anthropic   adaptive thinking (effort=medium) + web search
  gpt-5.6-search      OpenAI      reasoning_effort=medium + web search (medium context)
  gemini-3-retrieval  Google      thinking_budget=16384 + Google Search grounding
  sonar-deep-research Perplexity  always-on multi-step deep research
  grok-4.5-search     xAI         reasoning_effort=medium + web search

API keys are read from environment variables (see .env.example).

Contamination guard: all 5 models exclude EXCLUDED_SOURCE_DOMAINS from
retrieval — the claims are sourced from Lenz, so without this a
retrieval-capable model can find Lenz's own /c/<slug> verdict page and
parrot it back instead of forming an independent judgment. Enforced at
the API level per provider (see ``excluded_domains`` wiring in llm.py),
not via the prompt.
"""

from __future__ import annotations

from typing import Any

EXCLUDED_SOURCE_DOMAINS: tuple[str, ...] = ('lenz.io',)

PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    # Fable 5: thinking is always on ({'type': 'adaptive'} is the only accepted
    # explicit form); depth is set by effort. Requires 30-day data retention on
    # the org (ZDR orgs 400 on every request); safety classifiers may return
    # stop_reason='refusal' (surfaces as an empty-response RuntimeError).
    'claude-fable-5': {
        'provider': 'anthropic',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'api_model': 'claude-fable-5',
        # 64000: thinking is always on and counts toward max_tokens; keep
        # >=64K headroom so a hard claim with several web-search rounds can't
        # truncate the verdict even when thinking runs long.
        'min_max_tokens': 64000,
        'extra': {
            'thinking_budget': 16000,
            'web_search': True,
            'schema_with_thinking': True,
            'effort': 'medium',
            'request_timeout_ms': 300000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
        # Fallback: any failed claude-fable-5 call (refusal, timeout, rate
        # limit, ZDR-retention 400, etc.) retries once against Opus 4.8
        # instead of leaving an error row — see factcheck_solver in task.py.
        # Opus 4.8 is in AnthropicProvider._OMIT_TEMPERATURE_MODELS too (same
        # adaptive-thinking / no-temperature handling as Fable 5), so the
        # 'extra' shape below is deliberately identical to the primary entry.
        'fallback': {
            'provider': 'anthropic',
            'api_key_env': 'ANTHROPIC_API_KEY',
            'api_model': 'claude-opus-4-8',
            'min_max_tokens': 64000,
            'extra': {
                'thinking_budget': 16000,
                'web_search': True,
                'schema_with_thinking': True,
                'effort': 'medium',
                'request_timeout_ms': 300000,
                'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
            },
        },
    },
    'gpt-5.6-search': {
        'provider': 'openai-responses',
        'api_key_env': 'OPENAI_API_KEY',
        'api_model': 'gpt-5.6-sol',
        # 64000: bumped from 32000 (was gpt-5.5's setting) — Sol's reasoning +
        # web-search rounds share the same truncation risk that made
        # claude-fable-5 need 64K headroom; well within Sol's 128K output cap.
        'min_max_tokens': 64000,
        'extra': {
            'web_search': {'search_context_size': 'medium'},
            'reasoning_effort': 'medium',
            'request_timeout_ms': 300000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
    },
    'gemini-3-retrieval': {
        'provider': 'gemini',
        'api_key_env': 'GEMINI_API_KEY',
        'api_model': 'gemini-3.1-pro-preview',
        'min_max_tokens': 8192,
        'extra': {
            'google_search_grounding': True,
            'schema_with_grounding': True,
            'thinking_budget': 16384,
            'request_timeout_ms': 300000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
    },
    'sonar-deep-research': {
        'provider': 'perplexity',
        'api_key_env': 'PERPLEXITY_API_KEY',
        'api_model': 'sonar-deep-research',
        # 16384: responses carry the <think>...</think> prefix in visible
        # content and deep-research outputs run long — 8192 risked truncation.
        'min_max_tokens': 16384,
        'extra': {
            'request_timeout_ms': 600000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
    },
    'grok-4.5-search': {
        'provider': 'xai-responses',
        'api_key_env': 'XAI_API_KEY',
        'api_model': 'grok-4.5',
        # grok-4.5's hard output cap is 30K tokens (was 32000 for grok-4.3) —
        # 28000 leaves safety margin instead of sitting at the exact boundary.
        'min_max_tokens': 28000,
        'extra': {
            'web_search': {},
            'reasoning_effort': 'medium',
            'request_timeout_ms': 300000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
    },
}
