# Frontier LLM Fact-Check Benchmark

Reproducibility package for **[paper title]**.

We evaluate five frontier LLMs on 1,000 real-world fact-checking claims sourced
from [Lenz](https://lenz.io) — a live fact-checking platform. Each model uses
maximum thinking depth, live web retrieval, and native JSON schema enforcement.

## Models

| Model | Provider | Thinking | Retrieval |
|---|---|---|---|
| claude-fable-5 | Anthropic | adaptive (effort=max) | web_search_20260209 |
| gpt-5.5-search | OpenAI | reasoning_effort=xhigh | web search (high context) |
| gemini-3-retrieval | Google | thinking_budget=32768 | Google Search grounding |
| sonar-deep-research | Perplexity | built-in multi-step | always-on deep research |
| grok-4.3-search | xAI | reasoning_effort=xhigh | web search |

## Dataset

`data/claims.json` — 1,000 claims sampled from Lenz. Carries `share_id`, an
opaque join key back to Lenz, but deliberately no gold verdict — this repo
never reads or persists a conclusion label anywhere, so it stays safe to
share even though this field ties a row back to a specific Lenz claim.
Claims were also selected with pairwise embedding distance >= 0.10, so claim
text remains a safe join key on its own if you ever strip `share_id` back
out. A JSON array of records:

```json
{"claim": "...", "date": "2026-03-14", "category": "Science", "share_id": "4890227d"}
```

`data/results.jsonl` — per-(claim × model) harvest outputs, one JSON object per line
(streamed as the run progresses; `data/results.json` is the same rows as a final array):

```json
{"claim": "...", "date": "2026-03-14", "category": "Science", "share_id": "4890227d", "model": "gpt-5.5-search", "verdict": "True", "reasoning": "...", "confidence": 9, "cost_eur": 0.0042, "latency_s": 18.3, "error": "", "raw_response": "...", "sources": ["https://..."]}
```

An erroring cell carries the same shape with `verdict`/`confidence`/`raw_response`
empty and `error` set to the exception message.

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

Every scored sample is also appended to `data/results.jsonl`, and the deduped
`data/results.json` snapshot is rewritten after each one — same row shape and
default path `harvest.py` uses, so `compare.py` (and the DB import) read
either harness's output interchangeably. Pass `-T out_path=...` to keep a
debugging run out of the shared file.

**Bundled runs** — one model, 50 claims at a time, inspecting
`data/results.json` between bundles. Inspect's `--limit` is a 1-indexed,
inclusive range:

```bash
inspect eval task.py -T model_key=claude-fable-5 --limit 1-50
inspect eval task.py -T model_key=claude-fable-5 --limit 51-100
inspect eval task.py -T model_key=claude-fable-5 --limit 101-150
```

**Backup path** — `harvest.py` is a plain script with no framework
dependency, kept as a fallback if Inspect itself is ever the blocker. Same
output format, same default path, safe to interleave with `task.py` runs:

```bash
python harvest.py --claims data/claims.json --models claude-fable-5 --offset 0 --limit 50
```

(`harvest.py`'s `--offset`/`--limit` are 0-indexed — different convention
from Inspect's 1-indexed `--limit` range above.)

Results should match `data/results.jsonl` up to model non-determinism.

## Prompt

Each model receives:

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
