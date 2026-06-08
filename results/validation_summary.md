# Human validation summary (2026-06-02)

Single annotator (lead author) + Mistral second-annotator proxy. Corrected dataset.

## Task A — paraphrase equivalence (negative control), n=100
- Human: 98/100 paraphrases are meaning-preserving (2 drifted: PN-006, PN-029, excluded).
- Mistral annotator: 100/100 equivalent; agrees with human on 98/100.
- Conclusion: the paraphrase_null negative control is valid. When judges reject these,
  it is genuine over-rejection, not correct rejection of drift.

## Task B — perturbation faithfulness spot-check, n=50 (10/operator)
- Human: 50/50 judged unfaithful → 100% agreement with intended labels, 10/10 every
  operator including the contestable citation_relocation and distractor_parroting.
- Validates that all 6 operators produce human-confirmed-unfaithful perturbations.

## Validity gate — balanced human gold (148 cells = 98 faithful + 50 unfaithful)
Per-judge agreement with HUMAN gold (defuses "grading judges with judges"):

| Judge | acc | κ vs human |
|---|---|---|
| claude_sonnet_4_6 | 80% | 0.59 |
| mistral_large_2 | 80% | 0.56 |
| faithjudge_style_sonnet | 78% | 0.56 |
| alignscore_large | 78% | 0.45 |
| ragas_style_sonnet | 72% | 0.25 |
| minicheck_flan_t5_large | 65% | 0.25 |
| hhem_2_1_open | 63% | 0.14 |

## Inter-annotator agreement (human vs Mistral second annotator)
- **Cohen's κ = 0.891, agreement 141/148 = 95.3%** on the balanced gold.
- Landis-Koch: "almost perfect." Single-annotator gold is corroborated by an
  independent model using the same rubric. The 7 disagreements are all cases where
  Mistral-annotator was more lenient than the human (3 citation_relocation, 2
  entity_swap, 2 numeric_drift).

## Over-rejection on human-confirmed-equivalent paraphrases (n=98)
| Judge | over-reject % |
|---|---|
| minicheck_flan_t5_large | 32% |
| faithjudge_style_sonnet | 30% |
| hhem_2_1_open | 24% |
| claude_sonnet_4_6 | 22% |
| mistral_large_2 | 20% |
| ragas_style_sonnet | 5% |
| alignscore_large | 3% |

Brittleness spans architectures; only continuous scorers are robust. Not LLM-specific.

## BONUS — the citation blind spot is partly a prompt-omission problem
Same model (Mistral), two prompts, on the 10 citation_relocation spot-check items:
- content-only **judge** prompt (as deployed): **4/10** caught
- attribution-aware **rubric** prompt: **7/10** caught

When asked to check attribution, Mistral catches nearly twice as many — but still
misses 3/10. So the blind spot is partly "judges aren't asked about attribution" and
partly a deeper limitation. This motivates the full attribution-aware-prompt
experiment (re-run all 100 citation_relocation cells × judges with both prompts,
~$3-5) to quantify the prompt effect ensemble-wide. Strongly recommended — it converts
the headline from "judges miss citations" to "judges miss citations, and prompting for
attribution only partly fixes it," which pre-empts the obvious reviewer objection.
