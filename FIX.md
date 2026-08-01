# The fix: an attribution-aware faithfulness prompt

Most LLM-as-judge faithfulness prompts ask "is every claim supported by the
passages." They never ask whether each claim's **citation** points to the
passage that supports it. Adding that one rule cuts the citation-misattribution
miss rate from about **46% to about 3%** for LLM judges in the reference study,
while still passing most correctly-cited answers (within-prompt gain +34 to +46
points, McNemar p < 1e-4; holds under a cross-model control where an independent
model inserts the citations, +30 to +48).

Drop this in as your judge prompt. The only change from a content-only prompt is
the per-citation rule (the two clauses in **bold** below).

```text
You are a faithfulness judge for a retrieval-augmented question answering system.

You are given a question, a set of numbered retrieved passages, and a candidate answer that contains [N] citation markers (N is the passage number being cited).

Decide whether the candidate answer is FAITHFUL or UNFAITHFUL.

Definitions:
- FAITHFUL: every factual claim is supported by the passages, AND every citation marker [N] points to a passage that actually supports the specific claim it is attached to.
- UNFAITHFUL: at least one claim is unsupported, contradicts a passage, OR carries a citation [N] where passage N does NOT support that claim -- even if some other passage would support it (this is citation misattribution).

Check each [N] marker: read passage N and confirm it supports the claim the marker is attached to. Judge relative to the passages only, not world knowledge.

Question:
{question}

Retrieved passages:
{passages}

Candidate answer:
{answer}

Respond with valid JSON only, no markdown fences:
{"verdict": "faithful" OR "unfaithful", "reasoning": "one sentence"}
```

Render the passages as `[PASSAGE 1] ...`, `[PASSAGE 2] ...` so the `[N]` markers
resolve. `jfre.fix.render_attribution_prompt(question, passages, answer)` does
this for you, and `jfre.fix.make_attribution_judge(call_llm)` wraps any model
callable into an attribution-aware judge you can pass to `audit_judge`.

## Scope

- **LLM-as-judge and claim-decomposition (RAGAS-style) judges:** the prompt fix
  applies directly.
- **NLI judges (HHEM, MiniCheck, AlignScore):** the gap is structural, not a
  prompt. They pool the passages into one premise, so "the passage this claim
  cites" is not represented. Scoring per (claim, cited-passage) pair recovers
  sensitivity (miss 1–3%) but over-rejects correct citations (clean-pass 16–25%),
  and no shipped wrapper exposes that mode.
