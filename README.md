# lenz-research

Reproducibility packages for [Lenz](https://lenz.io) research, published at
[lenz.io/research](https://lenz.io/research).

Each study under `studies/` is **self-contained**: the exact code,
provider configuration, corpus, and raw results that produced its paper
live in one directory and freeze together. New studies start by copying
the previous study's code — shared machinery gets factored out only once
several studies prove a stable common shape.

## Studies

| Study | Paper | Contents |
|---|---|---|
| [`studies/llm-disagreement/`](studies/llm-disagreement/) | [Beyond Benchmarks: Disagreement Among Frontier LLMs on Real-World Fact-Checks](https://lenz.io/research/llm-disagreement) | Inspect AI runner, five-model panel config, 1,000-claim corpus, 5,000-row harvest (JSON/JSONL/CSV) |

Each study's README carries its models, dataset documentation, reproduction
steps, and prompt. Reproduction runs execute from the study's own directory:

```bash
cd studies/llm-disagreement
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
inspect eval task.py -T model_key=claude-fable-5
```

## License

Dual-licensed:

- **Code** (each study's `*.py` files) — [MIT License](LICENSE).
- **Data** (each study's `data/` directory) — [Creative Commons Attribution
  4.0 International (CC BY 4.0)](studies/llm-disagreement/data/LICENSE).

If you use the data in research or other work, please attribute
Lenz (https://lenz.io). Citation details are in each study's README.
