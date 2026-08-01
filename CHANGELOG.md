# Changelog

All notable changes to `jfre` are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] — 2026-07-29

First public release. `jfre` is the runnable tool; **Fidecite** is the named
diagnostic it implements. The import package is `jfre` and stays `jfre`.

### Added

- `audit_judge(judge, seeds)` — audit any `(question, passages, answer) -> bool`
  faithfulness judge for the citation-attribution blind spot. Returns an
  `AuditResult` carrying the attribution false-negative rate, the clean-pass
  rate, and the skipped-seed count.
- `make_seed(question, passages, cited_answer, seed_id)` — build a seed from your
  own data. Needs at least two distinct `[N]` markers to be attackable.
- `relocate_citations(seed)` — the Fidecite attack on its own. Re-points every
  `[N]` marker with a fixed-point-free permutation, so each claim cites a passage
  that does not support it while staying supported by some other passage.
- `jfre.fix` — the drop-in fix. `ATTRIBUTION_AWARE_PROMPT` plus
  `make_attribution_judge(call_llm)`, which wraps any text-in/text-out model
  callable into an attribution-aware judge.
- `jfre.operators` — the six single-edit perturbation operators
  (`citation_relocation`, `entity_swap`, `numeric_drift`, `hedge_insertion`,
  `paraphrase_null`, `distractor_parroting`). `citation_relocation` is the
  attribution-only one.
- `jfre.judges` — the seven reference judges from the study, behind the
  `[judges]` extra.
- `jfre.seeds` — HotpotQA / ExpertQA / PubMedQA loaders, behind the `[data]` extra.
- `examples/audit_your_own_judge.py` and `examples/apply_the_fix.py`, both
  runnable with no API keys.
- PEP 561 `py.typed` marker.
- Packaging metadata: authors, project URLs, PyPI classifiers, `CHANGELOG.md`.

### Notes

- **Core install is dependency-free.** The BYO-judge audit path and the fix use
  only the standard library. `torch`, `transformers`, `anthropic`, `mistralai`,
  and `datasets` live in the `[judges]`, `[data]`, and `[env]` extras and are
  needed only to reproduce the study.
- Not yet on PyPI. Install from the repository URL; see the README.
- The metric name `citation-faithfulness` / `CitationFaithfulnessMetric` is
  shipped in third-party integrations and is treated as a stable public name.

[Unreleased]: https://github.com/rsrijith/joint-failure-rag-eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rsrijith/joint-failure-rag-eval/releases/tag/v0.1.0
