# jfre 0.1.0 — Fidecite

**The faithfulness check most RAG stacks trust is checking the wrong thing.** It
verifies that each claim is supported by the retrieved passages *somewhere*. It
does not verify that each claim traces to the *specific passage it cites*. An
answer that cites the wrong passage for every claim sails through as "faithful."

This release ships the audit, the attack, and the fix.

## Install

```bash
pip install jfre        # zero heavy dependencies; bring your own judge
```

## Audit your own judge in a few lines

```python
from jfre import make_seed, audit_judge

def my_judge(question, passages, answer) -> bool:   # True == "faithful"
    ...   # call YOUR RAGAS / HHEM / LLM-as-judge / custom metric

seeds = [
    make_seed(
        question="When did each event occur?",
        passages=["The first satellite launched in 1957.",
                  "The first Moon landing was in 1969."],
        cited_answer="The first satellite launched in 1957 [1]. "
                     "The first Moon landing was in 1969 [2].",
    ),
]
print(audit_judge(my_judge, seeds).summary())
```

A judge that checks attribution flips to "unfaithful" on the relocated answer. A
judge that still says "faithful" has the blind spot. The share of relocated
answers it still passes is its **attribution false-negative rate**.

## What's in it

- `audit_judge` / `make_seed` / `relocate_citations` — the audit and the Fidecite
  citation-relocation attack, standard library only.
- `jfre.fix` — `ATTRIBUTION_AWARE_PROMPT` and `make_attribution_judge(call_llm)`,
  the drop-in fix for LLM-as-judge and claim-decomposition judges.
- `jfre.operators` — six single-edit perturbation operators;
  `citation_relocation` is the attribution-only one.
- `jfre.judges` (extra `[judges]`) — the seven reference judges from the study.
- `jfre.seeds` (extra `[data]`) — HotpotQA / ExpertQA / PubMedQA loaders.
- `examples/` — two runnable scripts, no API keys needed.

Core install pulls no dependencies. `torch`, `transformers`, `anthropic`,
`mistralai`, and `datasets` are only needed to reproduce the study.

## The study

Seven deployed faithfulness judges audited across 1,363 adversarial judge-cells.
The findings, the pre-registered methodology, and the post-pilot reframing are in
[README.md](README.md), [METHODOLOGY.md](METHODOLOGY.md), and
[POST_HOC_PIVOT.md](POST_HOC_PIVOT.md). The fix and its NLI caveat are in
[FIX.md](FIX.md).

## Rank your judge

The [Fidecite leaderboard](https://github.com/rsrijith/fidecite-leaderboard)
ranks judges by attribution FNR. Submitting is a script run and a pull request;
no sign-up, nobody to ask.

## Contributing

Good first issues are open. `CONTRIBUTING.md` has the setup, which is
`pip install -e ".[dev]"` and `pytest`.

Code is MIT. The perturbation dataset carries mixed per-subset source licenses;
see [`dataset/NOTICES.md`](dataset/NOTICES.md).

---

<!-- PRE-PUBLISH CHECKS, delete before pasting into the GitHub release form:
     1. The `pip install jfre` line above is the POST-PYPI form. If PyPI upload
        has not happened, change it to:
          pip install "git+https://github.com/rsrijith/joint-failure-rag-eval.git"
     2. If the distribution ships as `fidecite` (PUBLICATION_NOTES/dist-name.md),
        change it to `pip install fidecite`. The import stays `import jfre`.
     3. The leaderboard link 404s until that repo is pushed. Drop the section or
        push the leaderboard first.
     4. Re-check the study numbers against the camera-ready table if the paper
        decision has landed since 2026-07-29. -->
