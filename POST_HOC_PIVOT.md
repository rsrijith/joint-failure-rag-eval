# Post-pilot reframing: original hypothesis vs what the data showed

**Date of reframing: 2026-05-28**, after Phase 2 cell 398/500 of the main pilot. See `METHODOLOGY.md` for the timestamped pre-registration committed on 2026-05-26 before any data was collected.

This document is the honest record of how the paper's claims changed after seeing the data, what we now claim instead, and why.

## What the pre-registration said

The pre-registered hypothesis was that single-edit perturbations would induce *joint failure* — the entire judge ensemble simultaneously returning "faithful" on a perturbation that is in fact unfaithful — at headline-grade rates. The pre-registered go threshold was:

> **GO criterion:** at least one operator with joint-failure rate ≥ 25% on the pilot.

The hypothesis behind this threshold was that production ensemble-voting strategies (e.g., majority vote, worst-case vote) would have a meaningful failure floor introduced by correlated cross-judge errors.

## What the data showed (pilot at cell 398/500, n = 942 fully-scored adversarial cells)

Joint-failure rate per operator, 7-judge ensemble:

| Operator | Joint failure | Attack universality vs independence baseline |
|---|---|---|
| distractor_parroting | 2.8% | +2.76pp |
| entity_swap | 0.7% | +0.70pp |
| hedge_insertion | 0.0% | +0.00pp |
| numeric_drift | 2.1% | +2.13pp |

**No operator clears the 25% threshold.** The pre-registered headline framing does not survive the data.

However, the data revealed three signals that the pre-registration did not anticipate:

### Signal 1: Quality gap between LLM-judges and NLI fact-checkers

Per-architecture false-negative rate (% of perturbations called "faithful" when the perturbation is in fact unfaithful):

| Operator | LLM-judge avg | NLI fact-checker avg (HHEM, MiniCheck) |
|---|---|---|
| distractor_parroting | 17.0% | 59.2% |
| entity_swap | 3.6% | 26.5% |
| hedge_insertion | 1.4% | 7.3% |
| numeric_drift | 6.1% | 41.8% |

LLM-judges (Claude Sonnet 4.6, Mistral Large 2, FaithJudge-style) consistently miss fewer unfaithful answers than NLI fact-checkers (HHEM-2.1-Open, MiniCheck-Flan-T5-L) on every operator. The architectural quality gap is large enough to matter for ensemble design.

### Signal 2: LLM-judge verdicts are correlated within-family

Pairwise Cohen's κ within the LLM-judge family (Claude Sonnet, Mistral Large 2, FaithJudge-style):

| Pair | κ |
|---|---|
| Claude Sonnet vs FaithJudge-style | +0.635 |
| Claude Sonnet vs Mistral Large 2 | +0.324 |
| FaithJudge-style vs Mistral Large 2 | +0.276 |

Average within-LLM-cluster κ is +0.412. The 3rd LLM-judge never flips the LLM-cluster majority verdict (0 of 942 cells). For majority-vote ensemble strategies, the 3rd LLM-judge adds no information.

### Signal 3: A small ensemble approximates the large one

Best-subset agreement with the majority verdict of the full 6-judge ensemble:

| Subset size | Best subset | Agreement |
|---|---|---|
| 2 | Claude Sonnet + MiniCheck | 91.7% |
| 3 | Claude Sonnet + HHEM + MiniCheck | 96.8% |
| 4 | Claude Sonnet + Mistral + HHEM + MiniCheck | 97.7% |

A 2-judge architecture-diverse ensemble (1 LLM + 1 NLI) captures 91.7% of the full ensemble's majority verdict. Adding judges beyond the 4th yields < 1pp of additional agreement.

### Auxiliary finding: AlignScore-large is non-discriminative on this task

AlignScore returns "faithful" on **100%** of clean answers and **90.3%** of perturbed answers. The ratio is 0.903 — AlignScore does not meaningfully distinguish between clean and perturbed answers on single-edit perturbations of short-answer RAG. We exclude AlignScore from the headline 6-judge ensemble and report it separately.

## What we now claim (the reframed paper)

The paper's three contributions become:

1. **Quality gap between LLM-judges and NLI fact-checkers.** Across single-edit perturbations of RAG answers, LLM-as-judge averages 1-17% FNR while NLI fact-checkers average 7-59% on the same perturbations. The gap is large and architecture-aligned.

2. **Within-family redundancy of LLM-judges.** Three LLM-judges from three different organizations produce verdicts correlated enough at κ = +0.412 that the 3rd judge never changes the majority vote. For majority-voting ensembles, three LLM-judges from three vendors deliver one judge's worth of signal.

3. **Architecture-diverse beats organization-diverse.** A 2-judge ensemble (1 LLM + 1 NLI fact-checker) recovers 91.7% of the 6-judge majority verdict. Production systems stacking 5+ judges of similar architecture pay for redundant signal.

## What we drop from the original claim

- The "joint failure at high rates across the ensemble" framing. Joint-failure rates are 0-3% — interesting but not headline-grade.
- The promise of citation_relocation as the differentiating 6th operator. It may still differentiate if data is collected next week, but the paper does not require it to land.
- The 8-judge ensemble (GLM-4.7 dropped due to Cerebras free-tier quota exhaustion). The reported ensemble is 6 or 7 judges depending on whether AlignScore is included.

## Provenance

- `METHODOLOGY.md` is the timestamped pre-registration (committed 2026-05-26).
- This document records the post-hoc reframing rationale (drafted 2026-05-28).
- All raw verdicts are committed in `results/preview_pilot/verdicts.jsonl` for independent re-analysis under any alternative framing.
- The 942-cell figure is the count of *fully-scored* adversarial cells (every judge in the headline ensemble has a verdict, perturbation rule passed). The pilot continues to run; numbers in the paper draft will reflect the final dataset.
