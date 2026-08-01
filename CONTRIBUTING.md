# Contributing to jfre / Fidecite

Contributions welcome, especially judge adapters. If you have a faithfulness
metric that `jfre` cannot audit yet, that is the highest-value thing you can add.

## Setup

```bash
git clone https://github.com/rsrijith/joint-failure-rag-eval.git
cd joint-failure-rag-eval
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

14 tests, under a second, no network and no API keys. If `pytest` is green you
have a working environment.

Only install `".[judges]"` or `".[data]"` if you are reproducing the study.
Those pull torch, transformers, and the API SDKs.

## The one rule that matters

**The core package has no dependencies and it stays that way.** `import jfre`,
`audit_judge`, and `jfre.fix` must work with nothing but the standard library. A
pull request that adds anything to `[project] dependencies` in `pyproject.toml`
will not be merged. Put it in an optional extra.

Concretely: if your adapter needs `openai`, add an extra, and import the SDK
*inside* the adapter function or module, not at the top of a module that
`jfre/__init__.py` reaches. Look at `jfre/judges/claude_judge.py` for the shape.

## What a judge is

Any callable with this signature:

```python
def my_judge(question: str, passages: list[str], answer: str) -> bool:
    """True means faithful."""
```

`passages` is a list of plain strings. The answer's `[N]` markers are 1-indexed
into that list. That is the whole interface. If your metric returns a score,
threshold it inside the adapter and document the threshold you chose.

## Adding a judge adapter

Note there are two judge shapes in the tree, and new adapters should use the
first.

- **The public `Judge` callable** above. This is what `audit_judge` takes and what
  every BYO user writes.
- **The study judges** in `jfre/judges/`, which predate the public API and use
  `score(seed, answer_to_judge, operator) -> JudgeVerdict` because the reproduction
  pipeline needs the raw score and per-operator bookkeeping. Only match this shape
  if you are adding a judge to the study run itself.

To add an adapter:

1. New file at `jfre/judges/<name>_judge.py`.
2. Export a factory returning a `Judge`, e.g.
   `def make_ragas_judge(threshold: float = 0.5) -> Judge`.
3. Import the heavy SDK lazily, inside the factory or the returned closure, so
   `import jfre` never pulls it.
4. Add the dependency to the `judges` extra in `pyproject.toml`.
5. Add a test that exercises the adapter against a stub client, so the suite stays
   offline. `tests/test_fix.py` shows the stub pattern.
6. Say in the docstring whether the metric is an LLM-as-judge, a
   claim-decomposition metric, or an NLI model. The NLI ones have a structural
   limit documented in `FIX.md`; do not paper over it.

## Style

- `ruff check .` and `ruff format .` before you push.
- Type hints on public functions. The package ships `py.typed`.
- No new runtime dependency in core. Yes, this is the rule twice.

## Two names that cannot change

- The import package is **`jfre`**.
- The metric name is **`citation-faithfulness`** / **`CitationFaithfulnessMetric`**.

Both are already referenced by third-party integrations. Renaming either breaks
them silently, so those names are frozen regardless of how the branding evolves.
The brand is Fidecite; that is a documentation-layer name.

## Reporting a result rather than code

If you audited your own judge and want the number on the record, you do not need
to touch this repo. Submit to the
[Fidecite leaderboard](https://github.com/rsrijith/fidecite-leaderboard): run one
script, open a pull request. No sign-up.

## Changing the study's findings

The numbers in `README.md`, `METHODOLOGY.md`, and `POST_HOC_PIVOT.md` are the
results of a pre-registered run. If you think one is wrong, open an issue with
the reproduction rather than editing the figure in a pull request.

## Licensing

Code contributions are MIT, matching `LICENSE`. Do not add dataset rows without
checking `dataset/NOTICES.md`; the perturbation set carries mixed per-subset
source licenses and each row keeps its origin's terms.
