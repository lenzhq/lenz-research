# lenz-research

Reproducibility packages for [Lenz](https://lenz.io) research, published at
[lenz.io/research](https://lenz.io/research). Each study ships its corpus,
raw results, and a frozen reference to the exact code that produced them;
the shared evaluation runner lives in [`harness/`](harness/).

## Layout

```
harness/                 shared evaluation harness (Inspect AI task, provider
                         pool, pricing) — run from the repo root
studies/
  llm-disagreement/      "Beyond Benchmarks: Disagreement Among Frontier LLMs
                         on Real-World Fact-Checks" — corpus, raw results,
                         and study README
```

## Studies

| Study | Paper | Contents |
|---|---|---|
| [`studies/llm-disagreement/`](studies/llm-disagreement/) | [Beyond Benchmarks: Disagreement Among Frontier LLMs on Real-World Fact-Checks](https://lenz.io/research/llm-disagreement) | 1,000-claim corpus, 5,000-row five-model harvest (JSON/JSONL/CSV) |

Each study's README carries its models, dataset documentation, reproduction
steps, and prompt. Reproduction runs execute from the repo root, e.g.:

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
inspect eval harness/task.py -T model_key=claude-fable-5
```

## License

Dual-licensed:

- **Code** (`harness/`) — [MIT License](LICENSE).
- **Data** (each study's `data/` directory) — [Creative Commons Attribution
  4.0 International (CC BY 4.0)](studies/llm-disagreement/data/LICENSE).

If you use the data in research or other work, please attribute
Lenz (https://lenz.io). Citation details are in each study's README.
