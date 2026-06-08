# Deployed faithfulness judges are content-support-only (evidence, 2026-06-03)

Substantiates the paper's framing claim: production RAG faithfulness judges score
whether a claim is supported by the retrieved context as a whole, and NONE inspects
whether the passage a response *cites* actually supports the claim. Verbatim/close
quotes of shipped prompts + scoring methods.

| Tool | What it checks | Evidence (shipped prompt/method) | URL |
|---|---|---|---|
| **RAGAS Faithfulness** | content support | Decompose answer into atomic statements; for each "verify if it can be inferred from the given context." Score = supported claims / total. No per-citation check. | docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ |
| **TruLens Groundedness** | content support | System prompt: "INFORMATION OVERLAP classifier; providing the overlap (entailment or groundedness) between the source and statement." Returns supporting sentences from SOURCE holistically. | trulens.org/reference/trulens/feedback/llm_provider/ (templates/rag.py) |
| **FaithJudge (Vectara)** | content support | Flags claims "unsupported or contradicted by the provided source information"; binary Consistent/Inconsistent over the source document. | arxiv.org/abs/2505.04847 ; github.com/vectara/FaithJudge |
| **DeepEval FaithfulnessMetric** | content support | "ONLY provide a 'no' answer if the retrieval context DIRECTLY CONTRADICTS the claims"; claims not in context → 'idk'. Score = non-contradicted / total. | github.com/confident-ai/deepeval/.../faithfulness/template.py |
| **HHEM-2.1-Open** | content support (by construction) | Scores (premise, hypothesis) consistency 0-1 over the full premise; no notion of per-citation attribution. | huggingface.co/vectara/hallucination_evaluation_model |

**Counter-check:** per-citation attribution checking exists only in research citation-eval
benchmarks (ALCE, TRUST-SCORE arXiv 2409.11242), NOT in the deployed faithfulness judges
practitioners drop into RAG eval suites. So no widely-deployed judge validates citation identity.

**Citable two-sentence claim:** "Every widely-deployed RAG faithfulness judge we examined —
RAGAS Faithfulness, TruLens Groundedness, DeepEval FaithfulnessMetric, Vectara HHEM-2.1-Open,
and FaithJudge — scores whether a claim is supported by (or contradicted by) the retrieved
context as a whole, and none inspects whether the passage a response cites actually supports
the claim it is attached to. Per-citation attribution checking exists only in research
citation-evaluation benchmarks (ALCE, TRUST-SCORE), not in the production faithfulness metrics
practitioners deploy."

This de-strawmans the headline: deployed judges are content-support-only by construction
across the whole field, so the assumption that they catch citation errors is field-wide and
unsupported by the shipped tools — not an artifact of our own prompt choice.
