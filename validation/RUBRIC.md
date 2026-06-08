# Human Validation Rubric

**Committed before labeling begins. Do not edit after you start labeling.**
**Date:** 2026-06-02

You are the sole annotator. Two tasks. Read this whole file before opening either
packet. Label in the `*_labels.csv` files. Do not open `_answer_key.json` until
both tasks are complete (it contains the intended labels and would bias you).

Work top to bottom. Do not skip ahead to see patterns. If an item is genuinely
ambiguous, label your best judgment and write why in the `notes` column. Ambiguous
items are signal, not noise.

---

## Task A — Paraphrase Equivalence (100 items)

**Question you are answering for each item:** Does Answer B convey the *exact same
factual content* as Answer A?

- **yes** = same facts, even if the wording, clause order, or voice (active/passive)
  differs. Pure restyling.
- **no** = Answer B adds a fact, drops a fact, or changes a fact relative to A.
  Facts include: named entities, numbers, dates, places, quantities, and
  qualifiers/scope ("all" vs "some", "always" vs "often", "caused" vs "associated
  with").

Edge guidance:
- Reordering or combining sentences with no content change → **yes**.
- Dropping a hedge or adding certainty ("may reduce" → "reduces") → **no** (scope changed).
- A synonym that preserves meaning ("film" → "movie") → **yes**.
- A synonym that subtly shifts meaning ("expert" → "scientist") → **no**.
- If B omits a detail that A stated (even a minor one) → **no**.

Why this task exists: paraphrase_null is the study's negative control. The
perturbation is *supposed* to preserve faithfulness. Your labels test whether the
paraphrases are actually meaning-preserving, which is the precondition for
interpreting any judge's behavior on them.

---

## Task B — Perturbation Faithfulness Spot-Check (50 items)

**Question you are answering for each item:** Is the Candidate Answer FAITHFUL or
UNFAITHFUL to the retrieved passages?

Core definition (judge relative to the passages, NOT your own world knowledge):

- **faithful** = every factual claim in the answer is supported by the retrieved
  passages, AND every citation marker `[N]`, if present, points to a passage that
  actually supports the claim it is attached to.
- **unfaithful** = at least one of:
  1. a claim is not supported by any passage, OR
  2. a claim contradicts a passage, OR
  3. a claim carries a citation `[N]` but passage N does not support that claim
     (misattribution), even if some *other* passage would, OR
  4. the answer leans on content from a passage that does not actually address the
     question (off-topic support).

Apply the definition literally and consistently. Specific cases you will hit:

- **A swapped name/number/date** that does not appear in any passage → unfaithful (case 1).
- **A citation `[N]`** where passage N is about a different topic than the claim →
  unfaithful (case 3). Read passage N before deciding. This is the central test —
  do not skim. A claim whose content is true and appears *somewhere* in the
  passages but is cited to the *wrong* passage is still unfaithful.
- **A hedge** ("possibly", "some sources suggest") added to a claim the passages
  state plainly: judge whether the hedge misrepresents the passages. If the passages
  assert X flatly and the answer says "X is possibly true", that misrepresents the
  source → lean unfaithful, but use judgment and note it.
- **Appended content that is verbatim from a passage but off-topic** for the
  question → unfaithful (case 4). Ask: does this sentence actually help answer the
  question, or is it lifted from an unrelated passage?
- **World knowledge:** ignore it. If a claim is true in reality but unsupported by
  the passages, it is unfaithful for this task.

Why this task exists: it validates that the automated operators produced the
intended label (perturbed = unfaithful). Your labels are the human gold the paper
compares the automated judges against. Where you and the intended label disagree,
that disagreement is a finding about the operator, not an error by you.

---

## After you finish both CSVs

Tell me. I will:
1. Score your Task A labels against the paraphrase generator's equivalence claims
   and against each judge's verdict (reveals whether "LLM judges over-reject
   paraphrases" is real over-rejection or correct rejection of drifted paraphrases).
2. Score your Task B labels against the intended labels and each judge's verdict
   (gives human-vs-judge agreement = the validity gate the reviewers asked for).
3. Report inter-rater-style agreement (human vs each judge) and flag any operator
   where your labels diverge from the intended label.
