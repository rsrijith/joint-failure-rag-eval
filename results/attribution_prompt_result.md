# Attribution-prompt experiment + control (citation_relocation) — n=250

3 LLM-as-judge judges, **250 accepted citation-bearing seeds** (scaled from the original
100 on 2026-06-06). Same model, same answer, same passages; the only change is the prompt.
"Clean" = correct [N] citations (the gold cited answer); "scrambled" = citation_relocation
perturbation (claims mis-attributed). Faithful-rate shown; for clean we want HIGH, for
scrambled we want LOW.

NOTE (2026-06-06 correction): an earlier version of this file reported content:clean as
49/50/41% — that row was a transcription error (it duplicated the content:scrambled row).
The values below are recomputed directly from the raw pilot `clean_cited` verdicts
(results/citation_relocation_pilot/verdicts.jsonl, operator=clean_cited, Claude-annotated)
and the attribution-prompt experiment/control verdicts.

| Judge | content:clean | content:scrambled | attribution:clean | attribution:scrambled |
|---|---|---|---|---|
| claude_sonnet_4_6 | 80% | 48% | 81% | 4% |
| mistral_large_2 | 80% | 50% | 67% | 4% |
| faithjudge_style_sonnet | 72% | 39% | 77% | 3% |

## Discrimination (clean − scrambled faithful-rate; can the judge tell correct from incorrect citations?)

| Judge | content prompt Δ | attribution prompt Δ |
|---|---|---|
| claude_sonnet_4_6 | +32 pp | +77 pp |
| mistral_large_2 | +30 pp | +63 pp |
| faithjudge_style_sonnet | +33 pp | +74 pp |

## What this establishes (controlled, n=250)

1. **The deployed content-support prompt discriminates only partially and misses about half
   of misattributions.** It passes most correctly-cited answers (72–80%) but still calls
   ~39–50% of misattributed answers faithful (content:scrambled). The within-prompt gap is
   ~30 pp, not zero: the judge is not blind to citation correctness, but it tolerates
   misattribution roughly half the time.

2. **The blind spot is a prompt-specification gap, not a model limitation.** The
   attribution-aware prompt roughly doubles the discrimination (Δ 63–77 pp): the same model
   passes 67–81% of correctly-cited answers and flags 96–97% of misattributed ones.
   Content-prompt scrambled-FNR 47/50/37% → attribution-prompt scrambled-FNR 4/4/3%
   (gains +44/+46/+34 pp; McNemar p = 1.3e-30 / 9.6e-35 / 2.5e-23; bootstrap 95% CIs exclude 0).

3. **This control rules out the strictness artifact.** The low scrambled-FNR under the
   attribution prompt is real detection, not "calls everything unfaithful": the same prompt
   still passes the large majority of correctly-cited answers.

4. **NLI fact-checkers (HHEM, MiniCheck, AlignScore) cannot be fixed this way.** They have no
   prompt and pool all passages into one premise, so they cannot verify per-citation
   alignment as shipped (citation_relocation FNR 61/70/87%). Per-citation reconfiguration
   recovers sensitivity (miss 1–3%) but rejects most correct citations (16–25% clean-pass) —
   structural, not a prompt fix.

## Cross-model control (Llama-annotated citations, n=63)

The same pattern holds when an independent annotator (Llama-3.3-70B) inserts the citations
(rules out Claude-judges-Claude circularity): content:clean 79/87/75% vs content:scrambled
49/51/33%; attribution:scrambled 5/3/3%; attribution gains +44/+48/+30 pp (every McNemar
p < 1e-4). Both annotators show the content prompt discriminates partially but misses about
half of misattributions, and the attribution prompt fixes it either way.

Data: results/citation_relocation_pilot/verdicts_attribution_prompt.jsonl (scrambled),
verdicts_attribution_clean.jsonl (clean), verdicts_crossmodel_attribution.jsonl (Llama),
verdicts.jsonl (content-prompt clean_cited + citation_relocation),
scripts/attribution_prompt_experiment.py, attribution_prompt_control.py, stats_citation.py.
