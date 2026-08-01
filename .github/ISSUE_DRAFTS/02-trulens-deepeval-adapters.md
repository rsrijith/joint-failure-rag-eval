Title: Add adapters for TruLens Groundedness and DeepEval Faithfulness
Labels: good first issue, help wanted, adapter

## What

Two of the most widely deployed RAG faithfulness metrics cannot be audited without
glue code:

- **TruLens** `Groundedness` / `groundedness_measure_with_cot_reasons`
- **DeepEval** `FaithfulnessMetric`

Both are exactly the "is this claim supported by the passages *somewhere*" check
the audit is about, so they are the judges people most want a number for, and
neither has an adapter.

## Why it matters

Every audited judge is one more row on the
[Fidecite leaderboard](https://github.com/rsrijith/fidecite-leaderboard). These two
would likely be the most-viewed rows on it, because they are what teams actually
run in CI.

## What to do

Two files, or one PR with both. They are independent, so take either.

1. `jfre/judges/trulens_judge.py` with `make_trulens_judge(...) -> Judge`.
2. `jfre/judges/deepeval_judge.py` with `make_deepeval_judge(...) -> Judge`.

For each:

- Map `(question, passages, answer)` onto the upstream API's argument names. Both
  libraries take the retrieved context as a list of strings, so the mapping is
  direct.
- Both return a **continuous score**, not a boolean. Threshold it in the adapter,
  make the threshold a keyword argument, and **document the default and why you
  chose it**. This is the one judgement call in the issue; state it in the
  docstring rather than burying it.
- Import the library lazily inside the factory. Add `trulens-eval` /
  `deepeval` to the `judges` extra. Core `dependencies` stays empty.
- Add a test against a stub scorer so the suite stays offline.

## Definition of done

- `pytest` green with neither library installed.
- The threshold is a parameter with a documented default.
- A one-line note in the docstring on whether the metric decomposes claims, since
  that determines whether the prompt fix in `FIX.md` can help it or whether the
  limit is structural.

## Nice to have

Run the adapter over the Fidecite set and submit the resulting FNR to the
leaderboard. That is a separate PR in a separate repo and needs no permission.

## Pointers

- `jfre/judges/ragas_judge.py` — the existing claim-decomposition adapter, the
  nearest analogue to DeepEval.
- `jfre/judges/hhem_judge.py` — an existing score-thresholding adapter.
- `CONTRIBUTING.md` — the no-core-dependency rule and the stub-test pattern.
