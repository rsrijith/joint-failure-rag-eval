# Methodology: Cross-Judge Joint Failure of RAG Faithfulness Evaluators Under Single-Edit Attacks

**Status:** Pre-experiment methodology. Posted before running the full evaluation as a timestamped record of the design.
**Last updated:** 2026-05-26 (target push date: 2026-06-04)
**Code & artifacts:** This repository — full perturbation pipeline + judge evaluation harness + computed metrics will be released here post-acceptance.

> **Anonymization note.** During the GroundLM 2026 (EMNLP) double-blind review window, this repository is hosted anonymously. The methodology document is the public record of the planned design; data, code, and computed results are added after submission.

---

## Research question

Production RAG (retrieval-augmented generation) systems increasingly stack multiple automatic faithfulness judges — running RAGAS, HHEM, MiniCheck, AlignScore, FaithJudge, and LLM-as-judge variants from multiple organizations in parallel and trusting the majority verdict, the worst-case verdict, or a learned aggregator. This stacking implicitly assumes the judges fail independently.

We measure the *joint-failure rate* across **8 deployed judges**: the fraction of perturbed RAG answers (in fact unfaithful) on which *every* judge in the ensemble returns a faithful verdict. We show that joint-failure rate, pairwise Cohen's κ across judges, and "attack universality" per perturbation operator are distinct and operationally relevant statistical objects that none of GroUSE, FaithJudge, BUMP, RAGTruth, or FaithBench currently compute.

The hypothesis: single-edit perturbations designed to exploit specific judge-architecture shortcuts induce correlated failure across the ensemble, breaking the independence assumption at non-trivial rates.

---

## Perturbation suite (6 single-edit operators)

Operators 1 and 2 adapt Ma et al.'s BUMP (ACL 2023) Entity and Circumstance error categories to the RAG setting. Operators 3–6 are novel to the RAG faithfulness evaluation context and are the load-bearing novelty of this work.

Each perturbation is gated through an operator-specific automated rule. Perturbations failing the rule are discarded before any judge runs.

### 1. Entity swap *(BUMP-adapted)*
- **Edit:** Replace one content noun with a plausible same-class alternative drawn from the same context window.
- **Targets:** NLI-based judges (HHEM, MiniCheck, AlignScore) that may carry over entailment from surrounding faithful sentences.
- **Automated rule:** NER diff returns exactly one entity changed; both entities share the same coarse NER label.
- **Seed filter:** Wikipedia-derived seeds excluded from this operator to prevent world-knowledge-detectable swaps (Einstein ↔ Newton inflates accuracy for reasons unrelated to faithfulness reasoning — GroUSE §4.2 precedent).

### 2. Numeric drift *(BUMP-adapted)*
- **Edit:** Perturb a numeric value (e.g., 23% → 32%, 2019 → 2018).
- **Targets:** Claim-decomposition judges (RAGAS) that decompose claims into atoms but may not catch small numeric mismatches.
- **Automated rule:** the only changed token between original and perturbed is a parseable numeric literal; non-numeric tokens are identical.

### 3. Hedge insertion *(novel)*
- **Edit:** Insert hedges ("possibly," "according to some sources," "it has been reported that") in front of an unfaithful claim added to the answer.
- **Targets:** LLM-as-judge and FaithJudge — hedges may invoke "speculative content is permissible" interpretation.
- **Automated rule:** perturbed string contains at least one hedge token from a fixed lexicon AND a substring that is not entailed by the context (verified by a separate entailment check).

### 4. Citation-span relocation *(novel)*
- **Edit:** Move the cited span to a different (real, but irrelevant) passage in the retrieved context, while keeping the claim unchanged.
- **Targets:** Span-overlap and learned-discriminator judges (MiniCheck, TRACe, AlignScore) that may reward presence of a quoted match without verifying support.
- **Automated rule:** the quoted span in the perturbed answer matches a verbatim substring of a non-relevant retrieved passage (cosine similarity to query < 0.3) and not the originally-relevant passage.

### 5. Faithfulness-preserving syntactic paraphrase *(novel, negative control)*
- **Edit:** Rephrase the answer's syntax (active ↔ passive, clause reordering) without changing semantics.
- **Purpose:** Negative control. Any judge that flips on this perturbation has a robustness problem orthogonal to faithfulness.
- **Automated rule:** semantic equivalence (separate prompt) returns "equivalent" AND BLEURT score above a pre-registered threshold.

### 6. Distractor-passage parroting *(novel)*
- **Edit:** Insert verbatim text from a distractor passage (a retrieved passage not relevant to the question) into the answer, making a claim sourced from that distractor passage.
- **Targets:** Judges that reward verbatim overlap with any retrieved context, not just relevant context.
- **Automated rule:** inserted span exists verbatim in a retrieved passage with cosine similarity to query < 0.3.

---

## Judge set (8 deployed faithfulness evaluators)

All judges run on the same perturbed inputs. Model snapshots and versions pinned at experiment time and committed to this repo.

| Judge | Architectural family | Source |
|---|---|---|
| RAGAS faithfulness | Claim decomposition + LLM-judge | Es et al., EACL 2024 Demos; PyPI `ragas` v0.2.x |
| HHEM-2.1-Open | Fine-tuned NLI | Vectara HuggingFace model card |
| MiniCheck-Flan-T5-L | Fine-tuned Flan-T5 fact-checker | Tang et al., EMNLP 2024 (arXiv 2404.10774) |
| AlignScore-large | Fine-tuned alignment function | Zha et al., ACL 2023 |
| FaithJudge | Few-shot LLM-as-judge with curated hallucination examples | Tamber, Bao et al., EMNLP 2025 Industry (arXiv 2505.04847) |
| Claude 4 LLM-judge | Frontier proprietary LLM | Anthropic API, claude-opus-4-7 |
| Llama-3.3-70B LLM-judge | Frontier open-weights LLM | Meta, via Cerebras inference |
| Mistral Large 2 LLM-judge | Frontier proprietary LLM | Mistral API |

The four LLM-as-judge variants (FaithJudge, Claude, Llama, Mistral) span four distinct organizations and three model families. The three NLI-based judges (HHEM, MiniCheck, AlignScore) form an architecturally distinct cluster. This breakdown lets us compute Cohen's κ both within architectural family (do NLI judges share blind spots?) and across (do LLM-judges from different organizations correlate?).

Future-work judges not evaluated here: Prometheus 2, G-Eval, Trulens-Groundedness, Gemini 2.x.

---

## Seed data (500 total)

| Source | Count | License | Selection |
|---|---|---|---|
| HotpotQA (Yang et al., EMNLP 2018) | 250 | CC-BY-SA 4.0 | Stratified by hop count and distractor-passage presence |
| ExpertQA (Malaviya et al., NAACL 2024) | 250 | MIT | Stratified by expert domain |

Both sources are multi-passage retrieval-grounded, so all 6 operators apply uniformly.

**Faithful seed = gold human-written answer.** We use the dataset's gold answer as the faithful seed (not an LLM-generated answer). Seeds are faithful by construction; judges sanity-check rather than establish truth.

**Seed-faithful pre-filter:** ≥7 of the 8 judges agree the unperturbed gold answer is faithful. Examples that don't pass are discarded before perturbation.

**English-only.** Multilingual evaluation deferred to future work.

**Release-license inheritance.** Perturbations derived from HotpotQA inherit CC-BY-SA 4.0 (share-alike clause). Perturbations derived from ExpertQA inherit MIT. The released artifact carries a per-subset `NOTICES` file documenting upstream attributions and license assignments. Code repository is MIT throughout.

**RAGTruth was considered and dropped** as a seed source on 2026-05-26 after a license audit: ParticleMedia's MIT label covers their annotations but not the embedded MS MARCO (Microsoft non-commercial) and Yelp (non-sublicensable) passages, blocking redistribution. RAGTruth remains cited as a related-work natural-annotation baseline.

---

## Statistical framings (three, reported in parallel)

For each operator $o$, perturbation $a' = \text{op}_o(a)$ applied to seed $(c, a)$:

1. **Per-judge false-negative rate (marginals).** For each judge $j_i$, $\text{FNR}_i(o) = P(j_i(c, a') = \text{faithful} \mid a' \text{ is in fact unfaithful}, \text{op} = o)$.

2. **Joint-failure rate.** $\text{JFR}(o) = P(\bigcap_{i=1}^{8} \{j_i(c, a') = \text{faithful}\} \mid a' \text{ is in fact unfaithful}, \text{op} = o)$.

3. **Attack universality.** $U(o) = \text{JFR}(o) - \prod_{i=1}^{8} \text{FNR}_i(o)$. Positive values indicate judges fail more together than the independence baseline predicts. Negative values indicate sub-independence (less likely but possible).

4. **Pairwise Cohen's κ matrix (across judges, not annotators).** For each pair $(j_i, j_j)$, Cohen's κ on the binary faithful/unfaithful verdicts over the perturbed set. Reports the dependence structure directly, without the parametric independence assumption. Computed both pooled and stratified by architectural family (NLI cluster: HHEM/MiniCheck/AlignScore; LLM-judge cluster: FaithJudge/Claude/Llama/Mistral).

5. **Stratification.** All four statistics reported per (operator × seed-source) cell. Cells with fewer than 50 samples reported as descriptive only, not for headline claims.

---

## Validity gate (three layers)

| Layer | Coverage | Mechanism | Output |
|---|---|---|---|
| 1. Automated rule | All 3,000 perturbations | Operator-specific rule (per §1–6 above) | Per-operator rule-rejection rate |
| 2. Stratified human review | 600 perturbations (100 per operator, sampled across seed sources) | 2 annotators, binary label for "is the perturbed answer in fact unfaithful?" | Per-operator annotator-rejection rate; Cohen's κ between annotators |
| 3. Discussion round | All disagreements from layer 2 | Annotators discuss disagreements before lock (FaithBench §2.4 protocol) | Final per-operator validity-gated sample |

Perturbations failing any layer are excluded from headline numbers; rejection rate per operator reported as a transparency measure (§6 Table 3).

---

## Go/no-go thresholds (pre-committed before pilot starts)

Pilot: 50 examples × 2 cheapest operators (entity swap, numeric drift) × 8 judges, with author + 1 colleague labeling 30 examples.

**Continue with Candidate B (this paper) only if both:**
- At least one operator shows joint-failure ≥ 25% on the pilot.
- Inter-annotator agreement on the pilot 30 examples is Cohen's κ ≥ 0.6.

**Pivot to Candidate A ("Reward-hackable judges") if:**
- All operators show joint-failure < 10% on the pilot, OR
- Spot-check rejection rate > 50% on multiple operators (perturbation suite is broken).

**Reframe to a different venue (UncertaiNLP, Insights from Negative Results, AKBC) if:**
- Joint-failure is uniformly 10–25% (descriptive, not adversarial story).
- Judges disagree wildly even on clean answers (clean Fleiss' κ < 0.3 → calibration audit framing).
- One judge dominates (e.g., FaithJudge catches everything → comparison-paper framing).

---

## Timeline

| Phase | Dates | Output |
|---|---|---|
| Setup | 2026-05-31 → 2026-06-01 | License audit complete; judge versions pinned; 2 operators implemented |
| Pilot | 2026-06-02 → 2026-06-04 | 50-example pilot through 8 judges; 30 annotator pairs |
| **Methodology commit (this document)** | **2026-06-04 evening** | **Public timestamped record** |
| Build | 2026-06-07 → 2026-06-13 | Full 3,000-perturbation set generated and judged |
| Validate + analyze | 2026-06-14 → 2026-06-20 | 600-sample human review; tables and figures locked |
| Write + submit | 2026-06-21 → 2026-06-28 | Full 8-page draft |
| Submission | 2026-06-29 AoE | GroundLM 2026 @ EMNLP |

---

## Reproducibility commitments

- All judge inferences saved with seed, prompt, and exact model snapshot (HuggingFace commit SHA for HHEM/MiniCheck/AlignScore; API model ID + date for Claude/Llama-via-Cerebras/Mistral; pinned RAGAS version; FaithJudge prompt + few-shot pool committed to this repo).
- Automated-rule implementations (one Python module per operator).
- Perturbation seeds + accepted perturbations + per-judge verdicts released as a HuggingFace dataset post-acceptance.
- Annotator instructions + disagreement-resolution log committed to this repo.

---

## What this methodology does *not* cover (out of scope)

- Multi-edit perturbations (single-edit only, BUMP-style).
- Adaptive / judge-aware attacks (reserved for a planned follow-up on reward-hackable judges).
- Multilingual evaluation (deferred to future work).
- Judges trained or fine-tuned on RAG faithfulness data after the perturbation set is constructed (out of scope; would create a circular-evaluation problem).
- Inter-passage knowledge conflict resolution (reserved for a separate companion paper).
