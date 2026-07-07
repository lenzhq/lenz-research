# Frontier LLM Fact-Check Benchmark

Reproducibility package for **Beyond Benchmarks: Disagreement Among Frontier
LLMs on Real-World Fact-Checks**.

We evaluate five frontier LLMs on 1,000 real-world fact-checking claims sourced
from [Lenz](https://lenz.io) — a live fact-checking platform. Each model uses
medium thinking depth, live web retrieval, and native JSON schema enforcement.

## Models

| Model | Provider | Thinking | Retrieval |
|---|---|---|---|
| claude-fable-5 | Anthropic | adaptive (effort=medium) | web_search_20260209 |
| gpt-5.5-search | OpenAI | reasoning_effort=medium | web search (medium context) |
| gemini-3-retrieval | Google | thinking_budget=16384 | Google Search grounding |
| sonar-deep-research | Perplexity | built-in multi-step | always-on deep research |
| grok-4.3-search | xAI | reasoning_effort=medium | web search |

## Dataset

`data/claims.json` — 1,000 claims sampled from Lenz. Carries `verification_id`, an
opaque join key back to Lenz, but deliberately no gold verdict — this repo
never reads or persists a conclusion label anywhere, so it stays safe to
share even though this field ties a row back to a specific Lenz claim.
Claims were also selected with pairwise embedding distance >= 0.10, so claim
text remains a safe join key on its own if you ever strip `verification_id` back
out. A JSON array of records:

```json
{"claim": "...", "date": "2026-03-14", "category": "Science", "verification_id": "4890227d"}
```

`data/results.jsonl` — per-(claim × model) harvest outputs, one JSON object per line
(streamed as the run progresses; `data/results.json` is the same rows as a final array):

```json
{"claim": "...", "date": "2026-03-14", "category": "Science", "verification_id": "4890227d", "model": "gpt-5.5-search", "verdict": "True", "reasoning": "...", "confidence": 9, "cost_eur": 0.0042, "latency_s": 18.3, "error": "", "raw_response": "...", "sources": ["https://..."], "fallback_used": false, "fallback_model": ""}
```

An erroring cell carries the same shape with `verdict`/`confidence`/`raw_response`
empty and `error` set to the exception message.

**claude-fable-5 fallback:** any failed claude-fable-5 call (safety-classifier
refusal, timeout, rate limit, the ZDR-retention 400, etc.) retries once against
claude-opus-4-8 rather than being recorded as an error. The row still reports
`"model": "claude-fable-5"` (keeps the 5-model-per-claim panel shape intact),
but `fallback_used` is `true` and `fallback_model` is `"claude-opus-4-8"` —
filter on `fallback_used` before treating a row as a genuine Fable 5 answer.
`cost_eur`/`latency_s` include both the failed primary attempt and the
fallback attempt.

## Reproduce

Primary path is `task.py` via [Inspect AI](https://inspect.aisi.org.uk/) — one
model per invocation, with Inspect's own transcript viewer for stepping
through a run turn-by-turn:

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
inspect eval task.py -T model_key=claude-fable-5
inspect view
```

Every scored sample is upserted into `data/results.jsonl` (one row per
(claim, model) cell), and the deduped `data/results.json` snapshot is
rewritten after each one, so `compare.py` (and the DB import) always read a
current, duplicate-free file. Pass `-T out_path=...` to keep a debugging run
out of the shared file.

Inspect caps concurrent Model API calls at `--max-connections 10` by default —
that's the effective throttle on how fast a run burns through claims. Raise it
(e.g. `--max-connections 20`) if your provider's rate limits allow it, or pass
it explicitly to pin the behavior instead of relying on the default:

```bash
inspect eval task.py -T model_key=claude-fable-5 --max-connections 10
```

**Bundled runs** — one model, 50 claims at a time, inspecting
`data/results.json` between bundles. Inspect's `--limit` is a 1-indexed,
inclusive range:

```bash
inspect eval task.py -T model_key=claude-fable-5 --limit 1-50
inspect eval task.py -T model_key=claude-fable-5 --limit 51-100
inspect eval task.py -T model_key=claude-fable-5 --limit 101-150
```

**Resume** — re-running any range is safe and cheap: samples that already
have a successful row for that model in `out_path` are skipped without an
API call, while errored rows (refusals, timeouts, rate limits, quota
failures) are retried and their line replaced in place. So after a partial
failure, just re-run the full range:

```bash
inspect eval task.py -T model_key=claude-fable-5 --limit 1-1000
```

Results should match `data/results.jsonl` up to model non-determinism.

**Known caveat (Gemini)** — the google-genai SDK only supports
`exclude_domains` on its search-grounding tool in Vertex AI "Enterprise
Agent Platform" mode; with a plain Developer API key it raises a client-side
`ValueError` on every request. `gemini-3-retrieval` therefore runs *without*
the lenz.io domain-exclusion contamination guard the other four providers
enforce natively. A post-run audit of all grounded sources found zero
lenz.io citations.

## Prompt

Each model receives a **system prompt** and a **user prompt**. `{date}` and
`{claim}` are filled in per claim.

**System prompt:**

```
You are a fact-checker. Evaluate the truthfulness of the claim as of the stated date. Respond with JSON only.
```

**User prompt:**

```
Evaluate this claim as of {date}:
"{claim}"

Choose exactly one verdict from the scale below:
  "True" — the claim is accurate
  "Mostly True" — the claim is largely accurate with minor caveats or omissions
  "Mixed" — the claim has both accurate and inaccurate elements
  "Mostly False" — the claim is largely inaccurate with some basis in fact
  "False" — the claim is inaccurate

Respond with a JSON object containing exactly these fields:
  "reasoning": 2-4 sentences of claim analysis and verdict justification
  "verdict": one of the five labels above
  "confidence_level": your level of certainty in the verdict on a 1 to 10 integer scale (1 = completely uncertain, 10 = fully certain)
```
