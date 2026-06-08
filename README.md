# joint-failure-rag-eval

Adversarial perturbation suite and cross-judge analysis for RAG faithfulness evaluators. Code and data for an audit of seven deployed faithfulness judges, with a focus on **citation attribution** as a failure mode they share.

> See **[METHODOLOGY.md](METHODOLOGY.md)** for the timestamped pre-registration (operators, judges, statistical framings, go/no-go thresholds committed before data collection). See **[POST_HOC_PIVOT.md](POST_HOC_PIVOT.md)** for the post-pilot reframing — what the original hypotheses predicted, what the data actually showed, and how the headline shifted.

## What this project measures

Production RAG systems stack multiple faithfulness judges (RAGAS, HHEM, MiniCheck, AlignScore, FaithJudge, LLM-as-judge) and aggregate their verdicts by majority or worst-case vote, on the implicit assumption that judges from different families and organizations fail *independently*. We test that assumption empirically.

We apply six single-edit perturbation operators to fixed (question, passages, gold-answer) seeds and record each of seven deployed judges' verdicts, then compute per-judge false-negative rate, pairwise Cohen's κ, the ensemble joint-failure rate against an independence baseline, and ensemble-shrinkage curves.

The novel operator is **citation_relocation**: it deranges the `[N]` citation markers in a cited answer so every claim is attributed to a passage that does not support it, while the claim stays supported by *some* other passage. This isolates *attribution* faithfulness from *content* faithfulness — the dimension this audit shows deployed judges miss.

Seeds: HotpotQA + ExpertQA (main pilot, 500 accepted), ExpertQA-cited (citation_relocation, 250 accepted), and a PubMedQA biomedical replication. Target venue: **GroundLM 2026 @ EMNLP** (long paper, ACL Anthology archival, double-blind, deadline 2026-06-29 AoE).

## Headline findings

On 1,363 adversarial judge-cells (seven judges across the adversarial operators):

1. **Citation attribution is a blind spot across all seven judges.** Under the content-support prompts these judges ship with, every judge calls a large share of citation-misattributed answers "faithful" — per-judge false-negative rate **39% (FaithJudge, best) to 94% (RAGAS)**, and the three LLM judges miss about half. Relocated claims stay entailed by the passage set, so a content-support check passes them.

2. **The ensemble fails together.** On citation_relocation all seven judges unanimously pass the misattributed answer on **11.8% of cases (29/245)**, against a **3.2% independence baseline** (the product of the seven marginal pass-rates). The errors are correlated across architectures, so majority or worst-case voting does not rescue them.

3. **For LLM judges it is a prompt gap, not a model limit.** An attribution-aware prompt (which asks whether each claim is supported by the passage it cites) cuts the misattributed miss to **3–4%** while still passing most correctly-cited answers. Within-prompt gain **+34 to +46 points** (McNemar p < 1e-4), and it holds under a cross-model control where an independent Llama-3.3-70B inserts the citations (+30 to +48).

4. **For NLI judges it is structural.** HHEM, MiniCheck, and AlignScore pool the passages into a single premise, so "the passage this claim cites" is not represented. Scoring per (claim, cited-passage) pair recovers sensitivity (miss 1–3%) but over-rejects correct citations (clean-pass 16–25%), and no shipped wrapper exposes that mode.

Supporting results:

- **Judges cluster by architecture** (within-LLM κ 0.557, within-NLI κ 0.319, cross-architecture κ 0.165).
- **LLM-judge vendor diversity is largely redundant**: a third LLM judge from a third organization never flips the LLM-majority verdict (0 / 1358 cells).
- **A small architecture-diverse subset reproduces the full ensemble**: 2 judges match 86.7%, 3 judges 93.4%, and 4 judges 94.8% of the majority-of-seven.
- **Over-rejection of paraphrases**: judges reject 20–32% of human-confirmed meaning-preserving paraphrases (negative control), not specific to any one architecture.

The pre-registered hypothesis (some operator drives joint failure ≥ 25%) did **not** hold on the content operators (0–2.8%); citation_relocation is the operator that produces correlated joint failure. The reframing is documented in `POST_HOC_PIVOT.md`.

**Validity gate:** a human annotator labeled a stratified subset against a rubric committed before labeling; a second independent human annotator agrees at Cohen's κ = 0.920.

## Repository contents

| Path | Description |
|---|---|
| `jfre/operators/` | Six single-edit operators: entity_swap, numeric_drift, hedge_insertion, distractor_parroting, paraphrase_null (negative control), and citation_relocation (attribution-only). |
| `jfre/judges/` | Seven judge wrappers: Claude Sonnet 4.6, Mistral Large 2, FaithJudge-style (Sonnet + Vectara few-shot), RAGAS-style (Sonnet claim-decomposition), HHEM-2.1-Open, MiniCheck-Flan-T5-Large, AlignScore-large. |
| `jfre/seeds/` | HotpotQA, ExpertQA, and ExpertQA-cited seed loaders. |
| `scripts/run_preview_pilot.py` | Main pilot: per-source faithful pre-filter, perturbation generation, 7-judge scoring (resumable, skip-on-error, per-judge circuit breakers). |
| `scripts/run_citation_relocation_pilot.py` | citation_relocation pilot on ExpertQA-cited seeds. |
| `scripts/analyze_combined.py`, `scripts/kappa_adversarial_pool.py` | Per-judge FNR, joint-failure rate, pairwise κ matrix, ensemble shrinkage. |
| `scripts/attribution_prompt_experiment.py`, `scripts/attribution_prompt_control.py` | Attribution-aware prompt 2×2 (scrambled + clean control), including the cross-model annotator. |
| `scripts/f7_per_citation_nli.py` | Per-(claim, cited-passage) NLI ablation. |
| `scripts/stats_citation.py` | McNemar, bootstrap, and Wilson CIs for the attribution result. |
| `scripts/pubmedqa_*.py` | PubMedQA biomedical replication. |
| `scripts/second_annotator_*.py` | Second independent human annotator packets and scoring. |
| `scripts/full_audit.py` | Dataset integrity audit. |
| `results/` | Aggregated result files (`.txt`/`.md`). Raw verdicts, seeds, and perturbations (`.jsonl`) accompany the dataset release. |
| `validation/` | Human-validation rubric and annotation packets. |
| `METHODOLOGY.md` / `POST_HOC_PIVOT.md` | Pre-registration (timestamped) and post-pilot reframing. |
| `data/raw/` | Seed datasets (HotpotQA, ExpertQA, PubMedQA). Not redistributed; re-fetch from the upstream sources. |

## Reproducing

```bash
pip install -r requirements.txt          # or: pip install -e .
cp .env.example .env                      # add Anthropic + Mistral keys (no OpenAI)
# Local NLI judges (HHEM, MiniCheck, AlignScore) run on Apple Silicon GPU via JFRE_DEVICE=mps.
python3 scripts/run_preview_pilot.py            # main 5-operator pilot
python3 scripts/run_citation_relocation_pilot.py   # citation_relocation
python3 scripts/analyze_combined.py             # FNR, joint failure, kappa
python3 scripts/attribution_prompt_experiment.py && python3 scripts/attribution_prompt_control.py
```

Pilots are resumable (append-only JSONL keyed by seed/operator/judge). Seeds load deterministically (shuffle_seed 42). The two judge prompts (content-support and attribution-aware) are reproduced in the paper's appendix.

## Citing

A citation block will be added when the paper is accepted. In the meantime, the methodology committed on 2026-05-26 and the post-pilot reframing on 2026-05-28 are the timestamped record.

## License

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Perturbation dataset (when released):** mixed per-subset. HotpotQA-derived rows inherit CC-BY-SA 4.0 (share-alike clause); ExpertQA- and PubMedQA-derived rows remain under their source licenses. A per-row `NOTICES` file accompanies the released dataset.

Upstream datasets used as seeds:

- [HotpotQA](https://hotpotqa.github.io/) (Yang et al., EMNLP 2018) — CC-BY-SA 4.0
- [ExpertQA](https://github.com/chaitanyamalaviya/ExpertQA) (Malaviya et al., NAACL 2024) — MIT
- [PubMedQA](https://pubmedqa.github.io/) (Jin et al., EMNLP 2019) — MIT

RAGTruth (Niu et al., ACL 2024) was considered as a seed source and dropped after a license audit: the embedded MS MARCO and Yelp passages carry redistribution restrictions that propagate to derivative releases.
