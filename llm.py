"""Standalone LLM provider layer for the frontier fact-check benchmark.

Extracted from the Lenz pipeline (lenz/llm.py) with all Django/ORM
dependencies removed. Supports the five v1.1 benchmark providers:
Anthropic, OpenAI Responses, Perplexity, xAI Responses, and Gemini.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timeouts and defaults
# ---------------------------------------------------------------------------

ANTHROPIC_TIMEOUT = 300   # seconds
OPENAI_TIMEOUT = 300      # seconds
PERPLEXITY_TIMEOUT = 600  # seconds (sonar-deep-research is slow)
GEMINI_TIMEOUT_MS = 300_000  # milliseconds

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 800

# ---------------------------------------------------------------------------
# Model pricing (EUR per 1M tokens) — longest-prefix match
# ---------------------------------------------------------------------------

MODEL_PRICING_EUR: dict[str, tuple[float, float]] = {
    'claude-opus-4': (4.25, 21.25),
    'gpt-5.5': (4.25, 25.50),
    'gemini-3.1-pro': (1.70, 10.20),
    'sonar-deep-research': (1.70, 6.80),
    'grok-4.3': (1.06, 2.13),
}


def _lookup_pricing(model: str) -> tuple[float, float]:
    best_prefix = ''
    best_prices = (0.0, 0.0)
    for prefix, prices in MODEL_PRICING_EUR.items():
        if model.startswith(prefix) and len(prefix) > len(best_prefix):
            best_prefix = prefix
            best_prices = prices
    return best_prices


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _strip_additional_properties(schema: dict) -> dict:
    """Remove additionalProperties — Gemini's proto layer rejects it."""
    out: dict = {}
    for key, value in schema.items():
        if key == 'additionalProperties':
            continue
        if isinstance(value, dict):
            out[key] = _strip_additional_properties(value)
        elif isinstance(value, list):
            out[key] = [_strip_additional_properties(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


def _strip_anthropic_int_bounds(schema: dict) -> dict:
    """Remove min/max from integer fields — Anthropic structured output rejects them."""
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    is_integer = schema.get('type') == 'integer'
    for key, value in schema.items():
        if is_integer and key in ('minimum', 'maximum'):
            continue
        if isinstance(value, dict):
            out[key] = _strip_anthropic_int_bounds(value)
        elif isinstance(value, list):
            out[key] = [_strip_anthropic_int_bounds(v) if isinstance(v, dict) else v for v in value]
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class LLMProvider(ABC):
    def __init__(self, *, model: str, api_key: str, temperature: float, max_tokens: int, **extra: Any):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra = extra
        self.last_usage: dict | None = None

    @property
    def supports_json_schema(self) -> bool:
        return False

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 max_tokens: int | None = None) -> str:
        self.last_usage = None
        return self._complete_inner(system, user, json_schema=json_schema, max_tokens=max_tokens)

    @abstractmethod
    def _complete_inner(self, system: str, user: str, *, json_schema: dict | None = None,
                        max_tokens: int | None = None) -> str: ...

    def cost_eur(self) -> float:
        usage = self.last_usage or {}
        inp = int(usage.get('input_tokens', 0) or 0)
        out = int(usage.get('output_tokens', 0) or 0)
        price_in, price_out = _lookup_pricing(self.model)
        return inp * price_in / 1_000_000 + out * price_out / 1_000_000


# ---------------------------------------------------------------------------
# Anthropic — claude-opus-4-8 with adaptive thinking + web search
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    _OMIT_TEMPERATURE_MODELS = ('claude-opus-4-7', 'claude-opus-4-8')

    def _client(self):
        import anthropic
        timeout = int(self.extra.get('request_timeout_ms') or ANTHROPIC_TIMEOUT * 1000) / 1000
        return anthropic.Anthropic(api_key=self.api_key, timeout=timeout)

    @property
    def _omit_temperature(self) -> bool:
        return any(self.model.startswith(m) for m in self._OMIT_TEMPERATURE_MODELS)

    @property
    def _use_adaptive_thinking(self) -> bool:
        return any(self.model.startswith(m) for m in self._OMIT_TEMPERATURE_MODELS)

    @property
    def _thinking_budget(self) -> int | None:
        value = self.extra.get('thinking_budget')
        return int(value) if value is not None else None

    @property
    def _web_search(self) -> bool:
        return bool(self.extra.get('web_search', False))

    @property
    def _schema_with_thinking(self) -> bool:
        return bool(self.extra.get('schema_with_thinking', False))

    @property
    def _effort(self) -> str | None:
        value = self.extra.get('effort')
        return str(value) if value else None

    @property
    def supports_json_schema(self) -> bool:
        return self._schema_with_thinking or self._thinking_budget is None

    def _complete_inner(self, system: str, user: str, *, json_schema: dict | None = None,
                        max_tokens: int | None = None) -> str:
        client = self._client()
        kwargs: dict[str, Any] = {
            'model': self.model,
            'max_tokens': max_tokens or self.max_tokens,
            'system': system,
            'messages': [{'role': 'user', 'content': user}],
        }
        if not self._omit_temperature:
            kwargs['temperature'] = self.temperature

        output_config: dict[str, Any] = {}
        if self._effort:
            output_config['effort'] = self._effort

        if self._thinking_budget is not None:
            if self._use_adaptive_thinking:
                kwargs['thinking'] = {'type': 'adaptive'}
            else:
                kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': self._thinking_budget}
            if self._web_search:
                kwargs['tools'] = [{'type': 'web_search_20260209', 'name': 'web_search'}]
            if json_schema and self._schema_with_thinking:
                output_config['format'] = {
                    'type': 'json_schema',
                    'schema': _strip_anthropic_int_bounds(json_schema),
                }
        elif json_schema:
            output_config['format'] = {
                'type': 'json_schema',
                'schema': _strip_anthropic_int_bounds(json_schema),
            }

        if output_config:
            kwargs['output_config'] = output_config

        msg = client.messages.create(**kwargs)
        self.last_usage = {
            'input_tokens': msg.usage.input_tokens,
            'output_tokens': msg.usage.output_tokens,
        }
        texts = [block.text for block in msg.content if getattr(block, 'type', '') == 'text']
        text = '\n'.join(texts) if texts else ''
        if not text:
            raise RuntimeError(f'Anthropic returned empty response (model={self.model})')
        return text


# ---------------------------------------------------------------------------
# OpenAI Responses — gpt-5.5 with web search + reasoning
# ---------------------------------------------------------------------------

class OpenAIResponsesProvider(LLMProvider):
    _FIXED_TEMPERATURE_MODELS = {'gpt-5.4', 'gpt-5.5', 'gpt-5-nano', 'o1', 'o1-mini', 'o3', 'o3-mini'}

    @property
    def supports_json_schema(self) -> bool:
        return True

    def _client(self):
        import openai
        timeout = int(self.extra.get('request_timeout_ms') or OPENAI_TIMEOUT * 1000) / 1000
        return openai.OpenAI(api_key=self.api_key, timeout=timeout)

    @property
    def web_search(self) -> dict | None:
        value = self.extra.get('web_search')
        return value if isinstance(value, dict) else None

    @property
    def reasoning_effort(self) -> str | None:
        value = self.extra.get('reasoning_effort')
        return str(value) if value else None

    @property
    def _omit_temperature(self) -> bool:
        return any(self.model.startswith(m) for m in self._FIXED_TEMPERATURE_MODELS)

    def _complete_inner(self, system: str, user: str, *, json_schema: dict | None = None,
                        max_tokens: int | None = None) -> str:
        client = self._client()
        kwargs: dict[str, Any] = {
            'model': self.model,
            'instructions': system,
            'input': user,
            'max_output_tokens': max_tokens or self.max_tokens,
        }
        if not self._omit_temperature:
            kwargs['temperature'] = self.temperature
        if json_schema:
            kwargs['text'] = {
                'format': {'type': 'json_schema', 'name': 'structured_output', 'strict': True, 'schema': json_schema},
            }
        if self.web_search is not None:
            tool: dict = {'type': 'web_search'}
            if 'search_context_size' in self.web_search:
                tool['search_context_size'] = self.web_search['search_context_size']
            kwargs['tools'] = [tool]
            kwargs['tool_choice'] = 'auto'
        if self.reasoning_effort:
            kwargs['reasoning'] = {'effort': self.reasoning_effort}

        resp = client.responses.create(**kwargs)
        if getattr(resp, 'usage', None):
            self.last_usage = {
                'input_tokens': resp.usage.input_tokens,
                'output_tokens': resp.usage.output_tokens,
            }
        status = getattr(resp, 'status', 'unknown')
        output_text = getattr(resp, 'output_text', '') or ''
        if status != 'completed' or not output_text:
            incomplete = getattr(resp, 'incomplete_details', None)
            reason = getattr(incomplete, 'reason', status) if incomplete else status
            raise RuntimeError(f'OpenAI Responses incomplete (finish_reason={reason}, model={self.model})')
        return output_text


# ---------------------------------------------------------------------------
# Perplexity — sonar-deep-research
# ---------------------------------------------------------------------------

class PerplexityProvider(LLMProvider):
    @property
    def supports_json_schema(self) -> bool:
        return True

    def _client(self):
        import openai
        timeout = int(self.extra.get('request_timeout_ms') or PERPLEXITY_TIMEOUT * 1000) / 1000
        return openai.OpenAI(api_key=self.api_key, base_url='https://api.perplexity.ai', timeout=timeout)

    def _complete_inner(self, system: str, user: str, *, json_schema: dict | None = None,
                        max_tokens: int | None = None) -> str:
        client = self._client()
        messages = []
        if system and system.strip():
            messages.append({'role': 'system', 'content': system})
        messages.append({'role': 'user', 'content': user})
        kwargs: dict[str, Any] = {
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': max_tokens or self.max_tokens,
            'messages': messages,
        }
        if json_schema:
            kwargs['response_format'] = {
                'type': 'json_schema',
                'json_schema': {'name': 'verdict', 'schema': json_schema},
            }
        resp = client.chat.completions.create(**kwargs)
        if resp.usage:
            self.last_usage = {
                'input_tokens': resp.usage.prompt_tokens,
                'output_tokens': resp.usage.completion_tokens,
            }
        content = resp.choices[0].message.content or ''
        if not content:
            raise RuntimeError(f'Perplexity returned empty response (model={self.model})')
        return content


# ---------------------------------------------------------------------------
# xAI Responses — grok-4.3 with web search + reasoning
# ---------------------------------------------------------------------------

class XAIResponsesProvider(OpenAIResponsesProvider):
    def _client(self):
        import openai
        timeout = int(self.extra.get('request_timeout_ms') or OPENAI_TIMEOUT * 1000) / 1000
        return openai.OpenAI(api_key=self.api_key, base_url='https://api.x.ai/v1', timeout=timeout)


# ---------------------------------------------------------------------------
# Gemini — gemini-3.1-pro with search grounding + thinking
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    @property
    def search_grounding(self) -> bool:
        return bool(self.extra.get('google_search_grounding', False))

    @property
    def schema_with_grounding(self) -> bool:
        return bool(self.extra.get('schema_with_grounding', False))

    @property
    def supports_json_schema(self) -> bool:
        return (not self.search_grounding) or self.schema_with_grounding

    @property
    def thinking_budget(self) -> int | None:
        value = self.extra.get('thinking_budget')
        return int(value) if value is not None else None

    def _client(self):
        from google import genai
        timeout_ms = int(self.extra.get('request_timeout_ms') or GEMINI_TIMEOUT_MS)
        return genai.Client(api_key=self.api_key, http_options={'timeout': timeout_ms, 'retry_options': {'attempts': 2}})

    def _complete_inner(self, system: str, user: str, *, json_schema: dict | None = None,
                        max_tokens: int | None = None) -> str:
        from google.genai import types
        tools = [types.Tool(google_search=types.GoogleSearch())] if self.search_grounding else []
        config_kwargs: dict[str, Any] = {
            'system_instruction': system,
            'temperature': self.temperature,
            'max_output_tokens': max_tokens or self.max_tokens,
            'tools': tools or None,
        }
        if self.thinking_budget is not None:
            config_kwargs['thinking_config'] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        if json_schema and (not self.search_grounding or self.schema_with_grounding):
            config_kwargs['response_mime_type'] = 'application/json'
            config_kwargs['response_schema'] = _strip_additional_properties(json_schema)

        client = self._client()
        resp = client.models.generate_content(
            model=self.model, contents=user, config=types.GenerateContentConfig(**config_kwargs)
        )
        usage_meta = getattr(resp, 'usage_metadata', None)
        self.last_usage = {
            'input_tokens': getattr(usage_meta, 'prompt_token_count', 0) or 0,
            'output_tokens': (getattr(usage_meta, 'candidates_token_count', 0) or 0)
                + (getattr(usage_meta, 'thoughts_token_count', 0) or 0),
        }
        if not resp.text:
            reason = 'unknown'
            if resp.candidates:
                reason = getattr(resp.candidates[0], 'finish_reason', reason)
            raise RuntimeError(f'Gemini returned empty response (finish_reason={reason}, model={self.model})')
        text = resp.text
        if self.search_grounding:
            text = re.sub(r'<ctrl\d+>', '', text)
            text = re.sub(r'```tool_code\n.*?\n```', '', text, flags=re.DOTALL)
            text = text.strip()
            if not text:
                raise RuntimeError(f'Gemini returned only artifacts (model={self.model})')
        return text


# ---------------------------------------------------------------------------
# Provider registry + factory
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, type[LLMProvider]] = {
    'anthropic': AnthropicProvider,
    'openai-responses': OpenAIResponsesProvider,
    'perplexity': PerplexityProvider,
    'xai-responses': XAIResponsesProvider,
    'gemini': GeminiProvider,
}

_STANDARD_KEYS = {'provider', 'model', 'api_key_env', 'temperature', 'max_tokens', 'min_max_tokens'}


def build_provider(cfg: dict[str, Any], *, max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMProvider:
    """Instantiate an LLMProvider from a PROVIDER_CONFIG entry."""
    provider_name = cfg['provider']
    if provider_name not in _PROVIDERS:
        raise ValueError(f"Unknown provider '{provider_name}'. Known: {list(_PROVIDERS)}")

    api_key_env = cfg['api_key_env']
    api_key = os.environ.get(api_key_env, '')
    if not api_key:
        raise RuntimeError(f"Environment variable '{api_key_env}' is not set.")

    effective_max_tokens = max(max_tokens, cfg.get('min_max_tokens', 0))
    extra = {k: v for k, v in cfg.items() if k not in _STANDARD_KEYS}
    if 'extra' in extra:
        extra.update(extra.pop('extra'))

    return _PROVIDERS[provider_name](
        model=cfg['api_model'],
        api_key=api_key,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=effective_max_tokens,
        **extra,
    )
