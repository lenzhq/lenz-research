# Frontier LLM Fact-Check Benchmark

Reproducibility package for **[paper title]**.

We evaluate five frontier LLMs on 1,000 real-world fact-checking claims sourced
from [Lenz](https://lenz.io) — a live fact-checking platform. Each model uses
maximum thinking depth, live web retrieval, and native JSON schema enforcement.

## Models

| Model | Provider | Thinking | Retrieval |
|---|---|---|---|
| claude-opus-4-8 | Anthropic | adaptive (effort=max) | web_search_20260209 |
| gpt-5.5-search | OpenAI | reasoning_effort=xhigh | web search (high context) |
| gemini-3-retrieval | Google | thinking_budget=32768 | Google Search grounding |
| sonar-deep-research | Perplexity | built-in multi-step | always-on deep research |
| grok-4.3-search | xAI | reasoning_effort=xhigh | web search |

## Dataset

`data/claims.jsonl` — 1,000 claims sampled from Lenz, one JSON object per line:

```json
{"atomic_claim": "...", "domain": "Science", "conclusion_label": "True", "submission_date": "2026-03-14"}
```

`data/results.jsonl` — per-(claim × model) harvest outputs:

```json
{"share_id": "...", "model": "gpt-5.5-search", "verdict": "True", "reasoning": "...", "confidence": 9, "agrees_with_lenz": true, "cost_eur": 0.0042, "latency_s": 18.3}
```

## Reproduce

```bash
pip install anthropic openai google-genai
cp .env.example .env   # fill in your API keys
python harvest.py --claims data/claims.jsonl --out data/my_results.jsonl
```

Results should match `data/results.jsonl` up to model non-determinism.

## Prompt

Each model receives:

```
Evaluate this claim as of {date}:
"{claim}"

Choose exactly one verdict from the scale below:
  "True"         — the claim is accurate and well-supported by evidence
  "Mostly True"  — the claim is largely accurate with minor caveats or omissions
  "Mixed"        — the claim has both accurate and inaccurate elements
  "Mostly False" — the claim is largely inaccurate with some basis in fact
  "False"        — the claim is inaccurate or unsupported by evidence

Respond with a JSON object containing exactly these fields:
  "reasoning": 2-4 sentences of evidence-based analysis
  "verdict":   one of the five labels above
  "confidence_level": an integer from 1 to 10
```
