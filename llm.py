"""LLM provider layer for the frontier fact-check benchmark.

Supports five providers: Anthropic, OpenAI Responses, Perplexity,
xAI Responses, and Gemini. No external dependencies beyond the
provider SDKs.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

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
    'claude-fable-5': (8.50, 42.50),  # $10 / $50 — 2x Opus 4.8; thinking always on, billed as output
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
    if not best_prefix:
        # Silent €0.00 here means a renamed/unrecognized model's entire cost
        # column reads zero with no signal — loud enough to be seen in the
        # per-cell harvest.py output, not just a swallowed log line.
        print(f'WARNING: no pricing entry for model {model!r} — cost will read as €0.00')
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
        self.last_sources: list[str] = []

    @property
    def supports_json_schema(self) -> bool:
        return False

    @property
    def excluded_domains(self) -> list[str] | None:
        """Domains the provider's web search / grounding tool must never surface.

        Set via the ``excluded_domains`` extra. Each provider wires this into
        its own native domain-block mechanism (Anthropic/OpenAI/xAI/Gemini/
        Perplexity use different field names — see their ``_complete_inner``).
        API-enforced, unlike a prompt instruction.
        """
        value = self.extra.get('excluded_domains')
        return list(value) if value else None

    def complete(self, system: str, user: str, *, json_schema: dict | None = None,
                 max_tokens: int | None = None) -> str:
        self.last_usage = None
        self.last_sources = []
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
# Anthropic — claude-fable-5 with adaptive thinking + web search
# ---------------------------------------------------------------------------

class AnthropicProvider(LLMProvider):
    # Models that 400 on temperature/top_p/top_k AND on
    # thinking={'type': 'enabled', 'budget_tokens': N}. claude-fable-5:
    # thinking is always on — {'type': 'adaptive'} is the only accepted
    # explicit form ('disabled' also 400s, unlike Opus 4.7/4.8).
    _OMIT_TEMPERATURE_MODELS = ('claude-opus-4-7', 'claude-opus-4-8', 'claude-fable-5')

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

        if self._thinking_budget is not None:
            if self._use_adaptive_thinking:
                kwargs['thinking'] = {'type': 'adaptive'}
            else:
                kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': self._thinking_budget}

        # web_search (and the excluded_domains contamination guard riding on
        # it) is independent of thinking. Previously this was nested inside
        # `if self._thinking_budget is not None`, which meant removing
        # thinking_budget from a provider config would silently also drop
        # retrieval AND the domain-block guard with no error — a model could
        # then rediscover its own source claims via search with nothing
        # stopping it.
        if self._web_search:
            tool: dict[str, Any] = {'type': 'web_search_20260209', 'name': 'web_search'}
            if self.excluded_domains:
                tool['blocked_domains'] = self.excluded_domains
            kwargs['tools'] = [tool]

        output_config: dict[str, Any] = {}
        if self._effort:
            output_config['effort'] = self._effort
        # supports_json_schema already encodes "format is safe to send" —
        # thinking off, or schema_with_thinking explicitly opted in.
        if json_schema and self.supports_json_schema:
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
        sources: list[str] = []
        texts: list[str] = []
        for block in msg.content:
            btype = getattr(block, 'type', '')
            if btype == 'text':
                texts.append(block.text)
            elif btype == 'web_search_tool_result':
                for item in (getattr(block, 'content', None) or []):
                    url = getattr(item, 'url', None)
                    if url and isinstance(url, str) and url not in sources:
                        sources.append(url)
        self.last_sources = sources
        text = '\n'.join(texts) if texts else ''
        if not text:
            # Include stop_reason: Fable 5's safety classifiers return an empty
            # response with stop_reason='refusal' — downstream grading keys off
            # the word 'refusal' in this message to count abstains correctly.
            stop_reason = getattr(msg, 'stop_reason', 'unknown') or 'unknown'
            raise RuntimeError(f'Anthropic returned empty response (finish_reason={stop_reason}, model={self.model})')
        return text


# ---------------------------------------------------------------------------
# OpenAI Responses — gpt-5.5 with web search + reasoning
# ---------------------------------------------------------------------------

class OpenAIResponsesProvider(LLMProvider):
    _FIXED_TEMPERATURE_MODELS = {'gpt-5.4', 'gpt-5.5', 'gpt-5-nano', 'o1', 'o1-mini', 'o3', 'o3-mini'}

    # OpenAI's web_search tool names its domain-blocklist field
    # `blocked_domains`. xAI's Responses API mirrors this tool shape but
    # calls the same field `excluded_domains` — XAIResponsesProvider overrides.
    _domain_filter_key: str = 'blocked_domains'

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
            if self.excluded_domains:
                tool['filters'] = {self._domain_filter_key: self.excluded_domains}
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
        sources: list[str] = []
        for item in (getattr(resp, 'output', None) or []):
            if getattr(item, 'type', '') == 'message':
                for block in (getattr(item, 'content', None) or []):
                    for ann in (getattr(block, 'annotations', None) or []):
                        url = getattr(ann, 'url', None)
                        if url and isinstance(url, str) and url not in sources:
                            sources.append(url)
        self.last_sources = sources
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
        if self.excluded_domains:
            # search_domain_filter isn't a recognized kwarg on the openai-python
            # client's typed create() signature — it must go through extra_body,
            # which the SDK merges into the raw request body verbatim.
            # Hyphen-prefix denylists a domain.
            kwargs['extra_body'] = {'search_domain_filter': [f'-{d}' for d in self.excluded_domains]}
        resp = client.chat.completions.create(**kwargs)
        if resp.usage:
            self.last_usage = {
                'input_tokens': resp.usage.prompt_tokens,
                'output_tokens': resp.usage.completion_tokens,
            }
        self.last_sources = list(getattr(resp, 'citations', None) or [])
        content = resp.choices[0].message.content or ''
        if not content:
            raise RuntimeError(f'Perplexity returned empty response (model={self.model})')
        return content


# ---------------------------------------------------------------------------
# xAI Responses — grok-4.3 with web search + reasoning
# ---------------------------------------------------------------------------

class XAIResponsesProvider(OpenAIResponsesProvider):
    # xAI's web_search tool names its domain-blocklist field
    # `excluded_domains`, not OpenAI's `blocked_domains`.
    _domain_filter_key: str = 'excluded_domains'

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
        tools = []
        if self.search_grounding:
            # No exclude_domains: the google-genai SDK hard-rejects it
            # client-side (ValueError, before the request is sent) unless
            # the client is in Vertex AI "Enterprise Agent Platform" mode —
            # this repo authenticates with a plain Developer API key, so the
            # contamination guard the other 4 providers get natively isn't
            # available here. See README for the tradeoff.
            tools = [types.Tool(google_search=types.GoogleSearch())]
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
        if self.search_grounding:
            grounding_sources: list[str] = []
            for cand in (getattr(resp, 'candidates', None) or []):
                gm = getattr(cand, 'grounding_metadata', None)
                for chunk in (getattr(gm, 'grounding_chunks', None) or []):
                    web = getattr(chunk, 'web', None)
                    uri = getattr(web, 'uri', None)
                    if uri and isinstance(uri, str) and uri not in grounding_sources:
                        grounding_sources.append(uri)
            self.last_sources = grounding_sources
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

_STANDARD_KEYS = {'provider', 'api_model', 'api_key_env', 'temperature', 'max_tokens', 'min_max_tokens'}


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
