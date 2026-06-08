# Cross-model circularity control (2026-06-03)

The attribution-prompt experiment was re-run on citations produced by an INDEPENDENT
annotator — Meta Llama-3.3-70B via Groq, a different model family from every judge
(Claude / Mistral) — to test whether the attribution effect is an artifact of Claude
annotating citations that Claude (and Claude-backed FaithJudge/RAGAS) then judges.

63 usable Llama-cited seeds (>=2 citations). 2x2: content vs attribution prompt ×
clean vs scrambled (citation_relocation derangement), 3 LLM judges.

## 2x2 faithful-rate on Llama-annotated citations

| Judge | content:clean | content:scrambled | attribution:clean | attribution:scrambled |
|---|---|---|---|---|
| claude_sonnet_4_6 | 79% | 49% | 79% | 5% |
| mistral_large_2 | 87% | 51% | 71% | 3% |
| faithjudge_style_sonnet | 75% | 33% | 78% | 3% |

## Attribution Δ (clean − scrambled) — the headline, side by side

| Judge | Claude-annotated | Llama-annotated |
|---|---|---|
| claude_sonnet_4_6 | 82pp | 75pp |
| mistral_large_2 | 66pp | 68pp |
| faithjudge_style_sonnet | 77pp | 75pp |

## Conclusions

1. **Circularity ruled out.** The attribution Δ is essentially unchanged with an
   independent, different-family annotator (75/68/75 vs 82/66/77 pp). The effect is
   not Claude recognizing its own citation placements.

2. **The "Δ≈0 / content prompt is perfectly blind" sub-claim was annotator-specific
   and is DROPPED.** On Claude-annotated data content:clean≈content:scrambled (49≈49);
   on Llama-annotated data content:clean (79) > content:scrambled (49). The robust,
   annotator-independent fact is content:scrambled ≈ 49% under both annotators — the
   content prompt MISSES ~half of citation misattributions regardless of who annotated.

3. **Robust headline:** under a content-support prompt, faithfulness judges miss ~49%
   of citation misattributions; an attribution-aware prompt cuts the miss to 3-5% while
   still passing 71-79% of correct citations. Holds across annotators.

## Honest caveat
Llama and Claude annotate in slightly different styles: judges pass Llama's correct
citations more often (79%) than Claude's (49%) under the content prompt. This is
annotation-style variance, not text alteration (Llama preserves the gold answer text
modulo whitespace). Report the annotator-robust content:scrambled ≈ 49% and the
attribution Δ, not the fragile Claude-only Δ≈0.

Data: results/citation_relocation_pilot/verdicts_crossmodel_attribution.jsonl,
data/cache/expertqa_cited_llama.jsonl (annotator=llama-3.3-70b-versatile),
scripts/groq_annotate.py, gemini_attribution_experiment.py, compare_crossmodel.py.
