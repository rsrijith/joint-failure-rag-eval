---
license: other
license_name: mixed-per-row-see-notices
task_categories:
- text-classification
language:
- en
tags:
- rag
- faithfulness
- hallucination-detection
- attribution
- citation
- llm-evaluation
pretty_name: Fidecite — Citation-Attribution Faithfulness Set
size_categories:
- n<1K
---

# Fidecite — Citation-Attribution Faithfulness Set

> **Staging note (not yet published).** This card and `NOTICES.md` are staged in
> the repo. The row data is exported from the study artifact at release time so
> the per-source licensing below can be applied per row. Do not publish the rows
> without the matching `NOTICES` file.

A diagnostic set for the **citation-attribution** failure mode of RAG
faithfulness judges: each example pairs a correctly-cited answer with a
citation-relocated counterpart in which every `[N]` marker points to a passage
that does **not** support its claim (the claim stays supported by some *other*
passage). A judge that checks attribution must separate the two; a content-only
judge cannot.

## Fields

| field | type | description |
|---|---|---|
| `seed_id` | string | stable id of the source (question, passages, answer) seed |
| `source` | string | upstream seed dataset: `hotpotqa`, `expertqa`, or `pubmedqa` |
| `question` | string | the question |
| `passages` | list[string] | retrieved passages; `[N]` markers are 1-indexed into this list |
| `cited_answer` | string | the correctly-cited answer (label: faithful) |
| `relocated_answer` | string | the citation-relocated answer (label: unfaithful-attribution) |
| `permutation` | dict[int,int] | the marker remapping applied to produce `relocated_answer` |
| `n_citations` | int | number of distinct `[N]` markers relocated (≥ 2) |

## Intended use

Benchmark a faithfulness judge's **attribution false-negative rate**: the share
of `relocated_answer` rows it still labels faithful. The reference judges score
39%–94%. Use `jfre.audit.audit_judge` to run your own judge against it.

## Provenance and licensing

Rows are derived from three upstream seed datasets and **inherit their source
licenses per row** — see [`NOTICES.md`](NOTICES.md). HotpotQA-derived rows carry
CC-BY-SA 4.0 (share-alike); ExpertQA- and PubMedQA-derived rows remain under
their source (MIT) terms. RAGTruth was considered and dropped after a license
audit (embedded MS MARCO / Yelp passages carry redistribution restrictions).
