# Second Annotator — Instructions (read fully before labeling)

You are an INDEPENDENT second annotator for a study on RAG faithfulness evaluation.
Your labels will be compared against a first annotator's to measure inter-annotator
agreement. Please label on your own judgment. **Do not** discuss items with the first
annotator, and do not open any `_answer_key` file or the first annotator's label files
if you come across them — that would bias the comparison.

There are two short tasks, 25 items each (~30 minutes total). Label top to bottom; do
not skip ahead looking for patterns. If an item is genuinely ambiguous, give your best
judgment and write why in the `notes` column.

## Task A — Paraphrase Equivalence (taskA_paraphrase_packet.md -> taskA_labels.csv)

For each item you see Answer A (original) and Answer B (a paraphrase). Question:
**Does Answer B convey the exact same factual content as Answer A?**

- **yes** = same facts, even if wording, clause order, or voice (active/passive) differ.
  Pure restyling. Synonyms that preserve meaning ("film"->"movie") = yes.
- **no** = B adds, drops, or changes a fact relative to A. Facts include named entities,
  numbers, dates, places, quantities, and qualifiers/scope ("all" vs "some",
  "reduces" vs "may reduce"). Dropping a hedge or adding certainty = no. Omitting any
  detail A stated = no. A synonym that shifts meaning ("expert"->"scientist") = no.

Record `yes` or `no` in taskA_labels.csv under `equivalent_yes_no`.

## Task B — Perturbation Faithfulness (taskB_perturbation_packet.md -> taskB_labels.csv)

For each item you see a question, the retrieved passages, and a candidate answer (some
answers contain `[N]` citation markers pointing to passage N). Question:
**Is the candidate answer FAITHFUL or UNFAITHFUL to the passages?** Judge relative to the
passages only, NOT your own world knowledge.

- **faithful** = every factual claim is supported by the passages, AND every citation
  marker `[N]` points to a passage that actually supports the claim it is attached to.
- **unfaithful** = at least one of: (1) a claim is not supported by any passage; (2) a
  claim contradicts a passage; (3) a claim carries `[N]` but passage N does not support
  that claim (misattribution) — even if some OTHER passage would; (4) the answer leans on
  a passage that does not address the question (off-topic support).

Important for the `[N]` cases: read passage N before deciding. A claim whose content is
true and appears somewhere in the passages but is cited to the WRONG passage is still
unfaithful. A hedge ("possibly", "some sources suggest") added to a claim the passages
state plainly may misrepresent the source — use judgment and note it.

Record `faithful` or `unfaithful` in taskB_labels.csv under `faithful_yes_no`.

## When done

Send back the two filled CSVs (taskA_labels.csv, taskB_labels.csv). That's it — the
study author scores agreement from there.
