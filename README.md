# joint-failure-rag-eval

Adversarial perturbation suite and cross-judge agreement analysis for RAG faithfulness evaluators.

> See **[METHODOLOGY.md](METHODOLOGY.md)** for the timestamped pre-registration (operators, judges, statistical framings, go/no-go thresholds committed before data collection). See **[POST_HOC_PIVOT.md](POST_HOC_PIVOT.md)** for the post-pilot reframing of the paper's claims — what the original hypotheses predicted, what the data actually showed, and how the reported headline shifted.

## What this project measures

Production RAG systems stack multiple faithfulness judges (RAGAS, HHEM, MiniCheck, AlignScore, FaithJudge, Claude-as-judge, Mistral-as-judge, ...) and aggregate their verdicts. The implicit assumption is that judges from different families and organizations fail *independently*, so an ensemble vote is more robust than any single judge.

We test that assumption empirically. We apply five single-edit perturbation operators to ~500 (HotpotQA, ExpertQA) seeds and measure each judge's verdict on the perturbed answer, then compute:
- pairwise Cohen's κ across judges (do they agree on the same perturbations?),
- per-architecture false-negative rates (do LLM-judges and NLI-judges differ in quality?),
- ensemble-shrinkage curves (how small can the ensemble get before downstream agreement degrades?).

Target venue: **GroundLM 2026 @ EMNLP** (workshop, submission deadline 2026-06-29 AoE). Long paper, ACL Anthology archival, double-blind.

## Headline findings (preliminary, on ~942 fully-scored cells)

1. **LLM-as-judge consistently outperforms NLI fact-checkers on single-edit perturbations.** Across 4 adversarial operators, LLM-judge family (Claude Sonnet 4.6, Mistral Large 2, FaithJudge-style) averages 1-17% false-negative rate. NLI-judge family (HHEM-2.1-Open, MiniCheck-Flan-T5-L) averages 7-59% on the same perturbations. The architectural quality gap is large and consistent.

2. **Within-family verdict correlation makes LLM-judge ensembling redundant for majority voting.** Three LLM-judges from three different organizations agree closely enough that adding a 3rd judge never flips the majority verdict (0 of 942 cells in the current data).

3. **A small architecture-diverse ensemble matches a larger organization-diverse one.** A 2-judge ensemble (1 LLM + 1 NLI) reproduces 91.7% of the 6-judge majority verdict. A 3-judge ensemble (1 LLM + 2 NLI) reaches 96.8%. Judges beyond the 4th add less than 1pp of additional agreement.

4. **AlignScore-large is non-discriminative for short-answer RAG faithfulness.** AlignScore returns "faithful" on 100% of clean answers and 90% of perturbed answers — effectively constant. We exclude it from headline numbers and report it separately as a methodological note.

## Repository contents

| Path | Description |
|---|---|
| `jfre/operators/` | Five single-edit perturbation operators (entity_swap, numeric_drift, hedge_insertion, distractor_parroting, paraphrase_null) plus the sixth (citation_relocation) for attribution-only failures. |
| `jfre/judges/` | Wrappers for the seven judges: Claude Sonnet 4.6, Mistral Large 2, FaithJudge-style (Sonnet + Vectara few-shot), RAGAS-style (Sonnet claim-decomposition), HHEM-2.1-Open, MiniCheck-Flan-T5-Large, AlignScore-large. |
| `jfre/seeds/` | HotpotQA and ExpertQA seed loaders. |
| `scripts/run_preview_pilot.py` | Resumable orchestrator: per-source seed-faithful pre-filter, perturbation generation, 7-judge scoring with skip-on-error and per-judge circuit breakers. |
| `scripts/analyze_preview_pilot.py` | 5-operator per-judge FNR, joint-failure rate, pairwise κ matrix, attack universality. |
| `scripts/hypothesis_check_v2.py` | Rigorous validation of H-A (LLM-judge redundancy) and H-D (within-architecture diversity reversal). |
| `METHODOLOGY.md` | Pre-registered design document (timestamped). |
| `POST_HOC_PIVOT.md` | Post-pilot reframing rationale. |
| `data/raw/` | Seed datasets (HotpotQA, ExpertQA, AlignScore checkpoint cache). Not redistributed. |
| `results/preview_pilot/` | Pilot output (seeds.jsonl, perturbations.jsonl, verdicts.jsonl). |

## Citing

Citation block will land here when the paper is accepted. In the meantime, the methodology committed on 2026-05-26 and the post-pilot reframing on 2026-05-28 are the timestamped record.

## License

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Perturbation dataset (when released):** mixed per-subset. HotpotQA-derived rows inherit CC-BY-SA 4.0 (share-alike clause). ExpertQA-derived rows remain MIT. A per-row `NOTICES` file will accompany the released dataset.

Upstream datasets used as seeds:
- [HotpotQA](https://hotpotqa.github.io/) (Yang et al., EMNLP 2018) — CC-BY-SA 4.0
- [ExpertQA](https://github.com/chaitanyamalaviya/ExpertQA) (Malaviya et al., NAACL 2024) — MIT

RAGTruth (Niu et al., ACL 2024) was considered as a seed source and dropped after a license audit: the embedded MS MARCO and Yelp passages carry redistribution restrictions that propagate to derivative releases.
