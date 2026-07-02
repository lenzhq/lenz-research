"""Frontier-LLM provider configuration for the fact-checking benchmark.

Five models evaluated at full capacity — maximum thinking + live retrieval +
native JSON schema enforcement:

  claude-fable-5      Anthropic   adaptive thinking (effort=max) + web search
  gpt-5.5-search      OpenAI      reasoning_effort=xhigh + web search (high context)
  gemini-3-retrieval  Google      thinking_budget=32768 + Google Search grounding
  sonar-deep-research Perplexity  always-on multi-step deep research
  grok-4.3-search     xAI         reasoning_effort=xhigh + web search

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
        # 64000: thinking is always on and counts toward max_tokens; at
        # effort=max Anthropic recommends >=64K headroom so a hard claim with
        # several web-search rounds can't truncate the verdict.
        'min_max_tokens': 64000,
        'extra': {
            'thinking_budget': 16000,
            'web_search': True,
            'schema_with_thinking': True,
            'effort': 'max',
            'request_timeout_ms': 300000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
    },
    'gpt-5.5-search': {
        'provider': 'openai-responses',
        'api_key_env': 'OPENAI_API_KEY',
        'api_model': 'gpt-5.5',
        'min_max_tokens': 32000,
        'extra': {
            'web_search': {'search_context_size': 'high'},
            'reasoning_effort': 'xhigh',
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
            'thinking_budget': 32768,
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
    'grok-4.3-search': {
        'provider': 'xai-responses',
        'api_key_env': 'XAI_API_KEY',
        'api_model': 'grok-4.3',
        'min_max_tokens': 32000,
        'extra': {
            'web_search': {},
            'reasoning_effort': 'xhigh',
            'request_timeout_ms': 300000,
            'excluded_domains': list(EXCLUDED_SOURCE_DOMAINS),
        },
    },
}
