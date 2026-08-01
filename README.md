# jfre — Fidecite: audit your RAG faithfulness judge for the citation-attribution gap

**The faithfulness check most RAG stacks trust is only half-checking.** It verifies that each
claim is supported by the retrieved passages *somewhere*. It does **not** reliably verify that
each claim traces to the *specific passage it cites*. An answer that cites the wrong passage for
every claim gets called "faithful" roughly half the time. Test your judge in a few lines, and
apply the one-prompt fix.

**Fidecite** is the benchmark and the attack: the `citation_relocation` operator re-points every
`[N]` citation to a passage that does not support its claim, and the Fidecite leaderboard ranks
judges by how often they are fooled. The `jfre` package is the runnable tool; Fidecite is the
named diagnostic it implements.

```bash
# Zero heavy dependencies; bring your own judge. Not on PyPI yet, so install from source:
pip install "git+https://github.com/rsrijith/joint-failure-rag-eval.git"
# POST-RELEASE this becomes a one-liner:  pip install fidecite
```

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
# Attribution false-negative rate: 100% (relocated answers still passed) => BLIND SPOT
```

`audit_judge` relocates the `[N]` citation markers with a fixed-point-free permutation, so every
claim ends up attributed to a passage that does **not** support it (while staying supported by
some *other* passage). A judge that checks attribution flips to "unfaithful." A judge that still
says "faithful" has the gap. The share of relocated answers it still passes is its **attribution
false-negative rate**.

## What the study found

An audit of **seven deployed faithfulness judges** (Claude Sonnet 4.6, Mistral Large 2,
FaithJudge-style, RAGAS-style, HHEM-2.1-Open, MiniCheck, AlignScore) across 1,363 adversarial
judge-cells:

1. **Every judge tolerates a large share of citation misattribution.** Under the content-support
   prompts they ship with, per-judge false-negative rate on citation-misattributed answers runs
   **39% (FaithJudge, best) to 94% (RAGAS)**, and the three LLM judges still pass about half.
   **The content-support check discriminates only partially: it is not blind to citation
   correctness, but it lets misattribution through roughly half the time.** On the controlled
   n=250 run the LLM judges pass 72–80% of correctly-cited answers against 39–50% of
   misattributed ones, a within-prompt gap of about **+30 to +33 points** rather than zero.
2. **The ensemble fails together.** On citation_relocation all seven judges unanimously pass the
   misattributed answer on **11.8% of cases (29/245)**, against a **3.2% independence baseline**
   (the product of the seven marginal pass-rates). The errors are correlated across
   architectures, so majority or worst-case voting does not rescue them.
3. **For LLM judges it is a prompt gap, not a model limit.** An attribution-aware prompt (which
   asks whether each claim is supported by the passage it cites) cuts the misattributed miss to
   **3–4%** while still passing most correctly-cited answers, roughly doubling discrimination to
   **Δ +63 to +77 points** (within-prompt gain +34 to +46 points, McNemar p < 1e-4). It holds
   under a cross-model control where an independent Llama-3.3-70B inserts the citations (+30 to
   +48).
4. **For NLI judges it is the pooled-premise interface, not the model.** HHEM, MiniCheck, and
   AlignScore pool the passages into a single premise, so "the passage this claim cites" is not
   representable. Scoring per (claim, cited-passage) pair recovers sensitivity (miss 1–3%) but
   over-rejects correct citations (clean-pass 16–25%), and no shipped wrapper exposes that mode.
   So it is recoverable in principle, but not as a drop-in fix.

Supporting results:

- **Judges cluster by architecture** (within-LLM κ 0.557, within-NLI κ 0.319, cross-architecture
  κ 0.165).
- **LLM-judge vendor diversity is largely redundant**: a third LLM judge from a third
  organization never flips the LLM-majority verdict (0 / 1358 cells).
- **A small architecture-diverse subset reproduces the full ensemble**: 2 judges match 86.7%,
  3 judges 93.4%, and 4 judges 94.8% of the majority-of-seven.
- **Over-rejection of paraphrases**: judges reject 20–32% of human-confirmed meaning-preserving
  paraphrases (negative control), not specific to any one architecture.

The pre-registered hypothesis (some operator drives joint failure ≥ 25%) did **not** hold on the
content operators (0–2.8%); citation_relocation is the operator that produces correlated joint
failure. The reframing is documented in `POST_HOC_PIVOT.md`.

**Validity gate:** a human annotator labeled a stratified subset against a rubric committed
before labeling; a second independent human annotator agrees at Cohen's κ = 0.920.

See [METHODOLOGY.md](METHODOLOGY.md) for the timestamped pre-registration (operators, judges,
statistical framings, go/no-go thresholds committed before data collection), and
[POST_HOC_PIVOT.md](POST_HOC_PIVOT.md) for the post-pilot reframing — what the original
hypotheses predicted, what the data actually showed, and how the headline shifted.

## The fix (copy-paste)

For LLM-as-judge and claim-decomposition judges, the fix is one prompt change — ask, per claim,
whether the **cited** passage supports it. Full prompt in [FIX.md](FIX.md), or wire it in
directly:

```python
from jfre.fix import make_attribution_judge

def call_llm(prompt: str) -> str:
    ...   # return your model's raw text

judge = make_attribution_judge(call_llm)   # an attribution-aware jfre Judge
audit_judge(judge, seeds)                  # miss rate should drop toward ~3%
```

## Install

`jfre` is not on PyPI yet, so every install goes through the repo URL. Let
`JFRE=git+https://github.com/rsrijith/joint-failure-rag-eval.git`:

```bash
pip install "$JFRE"                    # BYO-judge audit + the fix (stdlib only)
pip install "fidecite[judges] @ $JFRE"  # + the seven reference judges (torch, SDKs)
pip install "fidecite[data] @ $JFRE"    # + HotpotQA / ExpertQA / PubMedQA loaders
```

<!-- POST-RELEASE: when the distribution is published, delete the JFRE line above
     and drop the ` @ $JFRE` suffixes, leaving:
       pip install fidecite
       pip install "fidecite[judges]"
       pip install "fidecite[data]"
     If the distribution ships under the name `fidecite` (see
     PUBLICATION_NOTES/dist-name.md), the import stays `import jfre` and only the
     three pip lines change to `pip install fidecite`, `"fidecite[judges]"`, etc. -->

## What this project measures

Production RAG systems stack multiple faithfulness judges (RAGAS, HHEM, MiniCheck, AlignScore,
FaithJudge, LLM-as-judge) and aggregate their verdicts by majority or worst-case vote, on the
implicit assumption that judges from different families and organizations fail *independently*.
We test that assumption empirically.

We apply six single-edit perturbation operators to fixed (question, passages, gold-answer) seeds
and record each of seven deployed judges' verdicts, then compute per-judge false-negative rate,
pairwise Cohen's κ, the ensemble joint-failure rate against an independence baseline, and
ensemble-shrinkage curves.

The novel operator is **citation_relocation**: it deranges the `[N]` citation markers in a cited
answer so every claim is attributed to a passage that does not support it, while the claim stays
supported by *some* other passage. This isolates *attribution* faithfulness from *content*
faithfulness — the dimension this audit shows deployed judges only partially check.

Seeds: HotpotQA + ExpertQA (main pilot, 500 accepted), ExpertQA-cited (citation_relocation, 250
accepted), and a PubMedQA biomedical replication. Target venue: **GroundLM 2026 @ EMNLP** (long
paper, ACL Anthology archival, double-blind, deadline 2026-06-29 AoE).

## Repository contents

| Path | Description |
|---|---|
| `jfre/audit.py`, `jfre/fix.py` | The bring-your-own-judge audit (`make_seed`, `audit_judge`) and the attribution-aware prompt fix (`make_attribution_judge`). Stdlib only. |
| `jfre/operators/` | Six single-edit operators: entity_swap, numeric_drift, hedge_insertion, distractor_parroting, paraphrase_null (negative control), and citation_relocation (attribution-only). |
| `jfre/judges/` | Seven judge wrappers: Claude Sonnet 4.6, Mistral Large 2, FaithJudge-style (Sonnet + Vectara few-shot), RAGAS-style (Sonnet claim-decomposition), HHEM-2.1-Open, MiniCheck-Flan-T5-Large, AlignScore-large. |
| `jfre/seeds/` | HotpotQA, ExpertQA, and ExpertQA-cited seed loaders. |
| `examples/` | `audit_your_own_judge.py` and `apply_the_fix.py` — runnable with no API keys. |
| `tests/` | Unit tests for the audit path, the fix path, and the citation_relocation operator. |
| `QUICKSTART.md` / `FIX.md` | BYO-judge walk-through, and the full attribution prompt. |
| `dataset/` | Perturbation-dataset staging and per-row source licensing (`NOTICES.md`). |
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

Pilots are resumable (append-only JSONL keyed by seed/operator/judge). Seeds load
deterministically (shuffle_seed 42). The two judge prompts (content-support and
attribution-aware) are reproduced in the paper's appendix.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Judge adapters for additional frameworks are the most
useful contribution; `.github/ISSUE_DRAFTS/` sketches a few good first tasks.

## Citing

A citation block is added when the paper is public; see [CITATION.cff](CITATION.cff). In the
meantime the methodology committed on 2026-05-26 and the post-pilot reframing on 2026-05-28 are
the timestamped record.

## License

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Perturbation dataset (when released):** mixed per-subset. HotpotQA-derived rows inherit
  CC-BY-SA 4.0 (share-alike clause); ExpertQA- and PubMedQA-derived rows remain under their
  source licenses. A per-row [`dataset/NOTICES.md`](dataset/NOTICES.md) accompanies the released
  dataset. **The share-alike clause propagates, so HotpotQA-derived rows cannot be bundled inside
  this MIT repository** — see `dataset/NOTICES.md` for the two approaches that do work.

Upstream datasets used as seeds:

- [HotpotQA](https://hotpotqa.github.io/) (Yang et al., EMNLP 2018) — CC-BY-SA 4.0
- [ExpertQA](https://github.com/chaitanyamalaviya/ExpertQA) (Malaviya et al., NAACL 2024) — MIT
- [PubMedQA](https://pubmedqa.github.io/) (Jin et al., EMNLP 2019) — MIT

RAGTruth (Niu et al., ACL 2024) was considered as a seed source and dropped after a license
audit: the embedded MS MARCO and Yelp passages carry redistribution restrictions that propagate
to derivative releases.
