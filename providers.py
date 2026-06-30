"""Frontier-LLM provider configuration for the fact-checking benchmark.

Five models evaluated at full capacity — maximum thinking + live retrieval +
native JSON schema enforcement:

  claude-opus-4-8     Anthropic   adaptive thinking (effort=max) + web search
  gpt-5.5-search      OpenAI      reasoning_effort=xhigh + web search (high context)
  gemini-3-retrieval  Google      thinking_budget=32768 + Google Search grounding
  sonar-deep-research Perplexity  always-on multi-step deep research
  grok-4.3-search     xAI         reasoning_effort=xhigh + web search

API keys are read from environment variables (see .env.example).
"""

from __future__ import annotations

from typing import Any

PROVIDER_CONFIG: dict[str, dict[str, Any]] = {
    'claude-opus-4-8': {
        'provider': 'anthropic',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'api_model': 'claude-opus-4-8',
        'min_max_tokens': 32000,
        'extra': {
            'thinking_budget': 16000,
            'web_search': True,
            'schema_with_thinking': True,
            'effort': 'max',
            'request_timeout_ms': 300000,
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
        },
    },
    'sonar-deep-research': {
        'provider': 'perplexity',
        'api_key_env': 'PERPLEXITY_API_KEY',
        'api_model': 'sonar-deep-research',
        'min_max_tokens': 8192,
        'extra': {'request_timeout_ms': 600000},
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
        },
    },
}
