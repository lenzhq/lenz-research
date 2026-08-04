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
| gpt-5.6-search | OpenAI | reasoning_effort=medium | web search (medium context) |
| gemini-3-retrieval | Google | thinking_budget=16384 | Google Search grounding |
| sonar-deep-research | Perplexity | built-in multi-step | always-on deep research |
| grok-4.5-search | xAI | reasoning_effort=medium | web search |

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
{"claim": "...", "date": "2026-03-14", "category": "Science", "verification_id": "4890227d", "model": "gpt-5.6-search", "verdict": "True", "reasoning": "...", "confidence": 9, "cost_eur": 0.0042, "latency_s": 18.3, "error": "", "raw_response": "...", "sources": ["https://..."], "fallback_used": false, "fallback_model": ""}
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

**Windows:** set `PYTHONIOENCODING=utf-8` before running `inspect eval` /
`inspect view`. Without it, Inspect's terminal progress display can crash
with a `UnicodeEncodeError` on Windows' legacy console encoding (cp1252
can't render some of the Unicode spinner/glyph characters Rich uses) —
this can happen before a single sample is scored.

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

**Estimated cost** for a full 1,000-claim run, computed from this repo's own
observed per-claim averages (your actual cost will vary with claim mix and
model pricing changes):

| Model | Avg €/claim | Est. for 1,000 claims |
|---|---|---|
| claude-fable-5 | 0.18 | ~€180 |
| gpt-5.6-search | 0.12 | ~€122 |
| gemini-3-retrieval | 0.02 | ~€16 |
| sonar-deep-research | <0.01 | ~€2 |
| grok-4.5-search | 0.14 | ~€135 |
| **Total (all 5 models)** | | **~€455** |

Every scored sample is upserted into `data/results.jsonl` (one row per
(claim, model) cell), and the deduped `data/results.json` snapshot is
rewritten after each one, so the Lenz DB import always reads a
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

**Don't run two `inspect eval` invocations at the same time** (e.g. two
different models in separate terminals), even though each targets a
different `model_key`. Every invocation upserts into the *same*
`data/results.jsonl` / `data/results.json` (see `_append_and_resnapshot` in
`task.py`), and the write lock guarding that read-modify-write is an
in-process `threading.Lock` — it does nothing across two separate OS
processes. On Windows this reliably surfaces as a `PermissionError` on
`os.replace()` (`results.jsonl.tmp` -> `results.jsonl`) and can interrupt a
run after zero or only a few samples. Run models one at a time, or point
concurrent runs at different `-T out_path=...` files and merge afterward.

**Resume** — re-running any range is safe and cheap: samples that already
have a successful row for that model in `out_path` are skipped without an
API call, while errored rows (refusals, timeouts, rate limits, quota
failures) are retried and their line replaced in place. So after a partial
failure, just re-run the full range:

```bash
inspect eval task.py -T model_key=claude-fable-5 --limit 1-1000
```

Results should match `data/results.jsonl` up to model non-determinism.

**Contamination guard** — `excluded_domains` is passed to every provider
that supports it and confirmed present on every call (see `_complete_inner`
in `llm.py`). Two models honor it completely across all 1,000 claims
(Claude Fable 5, GPT-5.6). Gemini's search-grounding tool doesn't accept a
domain-exclusion parameter under a standard Developer API key (only under
Vertex AI's "Enterprise Agent Platform" mode, which this repo doesn't use),
so `gemini-3-retrieval` ran without this guard; its grounding sources are
also opaque redirect URLs rather than resolvable domains, so it isn't
independently auditable either way. The two agentic/deep-research models
(Sonar Deep Research, Grok 4.5) don't fully honor the exclusion at the
provider level: lenz.io still appears among returned sources on 24/1000 and
15/1000 calls respectively, including 10 rows where the source was Lenz's
own prior verdict on the exact claim being evaluated. In those 10 rows the
model's verdict matched Lenz's own verdict 6 times and diverged the other 4
(only once by more than one bucket on the five-point scale) — not proof of
independence, but no sign of simple copying either.

## Troubleshooting

**`SSL: CERTIFICATE_VERIFY_FAILED` / `certificate verify failed: unable to
get local issuer certificate`** — every provider call fails with this. This
means something on your machine is intercepting outbound HTTPS (common with
antivirus software that does TLS/web-shield scanning, or a corporate proxy)
and Python's default certificate bundle doesn't trust the interceptor's
injected root CA — even though a browser on the same machine works fine,
since browsers pick up OS-level trusted roots that Python doesn't use by
default. Fix: locate your interceptor's root CA certificate (e.g. Avast's
Web/Mail Shield writes one to disk locally), append it to `certifi`'s
bundle (`python -c "import certifi; print(certifi.where())"` to find it) or
a copy of it, and point Python at the combined file:

```bash
export SSL_CERT_FILE=/path/to/combined_ca_bundle.pem
export REQUESTS_CA_BUNDLE=/path/to/combined_ca_bundle.pem
```

Do **not** work around this by disabling certificate verification — that
removes real protection against a genuine man-in-the-middle, not just the
interceptor you already know about.

## Prompt

Each model receives a **system prompt** and a **user prompt**. `{date}` and
`{claim}` are filled in per claim. `{date}` is the date the claim was submitted
to Lenz, which pins each model to the claim's submission-time epistemic frame.

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

## License

This repository is dual-licensed:

- **Code** (the `*.py` files) is licensed under the [MIT License](LICENSE).
- **Data** (everything in [`data/`](data/)) is licensed under the
  [Creative Commons Attribution 4.0 International License (CC BY 4.0)](data/LICENSE).

If you use the benchmark data in research or other work, please attribute
Lenz (https://lenz.io):

> Jordanov, K., Yordanov, D., and Jordanova, Y. (2026). *Beyond Benchmarks:
> Disagreement Among Frontier LLMs on Real-World Fact-Checks.* Lenz.

**What ships.** `data/` is allow-listed in `.gitignore`: only `claims.json`,
`results.json`, `results.jsonl` and `LICENSE` are tracked. Local working files
in that directory are deliberately excluded — several carry Lenz's own verdict
labels, and publishing those would hand a reader the gold key the panel was
denied.
