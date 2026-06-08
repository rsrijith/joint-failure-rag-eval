"""Build a blind 50-item packet for a SECOND independent human annotator.

Stratified subset of the first annotator's gold so human-human agreement is
apples-to-apples: 25 Task-A (paraphrase equivalence) + 25 Task-B (perturbation
faithfulness, 5 per adversarial operator). Same format and questions as the
original packets; item_ids preserved for scoring against annotator 1.

Outputs to validation/second_annotator/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

VAL = Path("validation")
OUT = VAL / "second_annotator"
OUT.mkdir(parents=True, exist_ok=True)

answer_key = json.load(open(VAL / "_answer_key.json"))
op_of = {k: v["operator"] for k, v in answer_key.items()}


def parse_blocks(path):
    """Return {item_id: full_block_text} split on '## <ID>' headers."""
    text = Path(path).read_text()
    parts = re.split(r"\n## ", text)
    blocks = {}
    for p in parts[1:]:
        item_id = p.split("\n", 1)[0].strip()
        blocks[item_id] = "## " + p.rstrip()
    return blocks


pn = parse_blocks(VAL / "paraphrase_equivalence_packet.md")   # PN-001..PN-100
sp = parse_blocks(VAL / "perturbation_spotcheck_packet.md")   # SP-001..SP-050

# --- select 25 paraphrase items, evenly spread across PN-001..PN-100 ---
pn_ids = sorted(pn, key=lambda k: int(k.split("-")[1]))
sel_pn = pn_ids[::4][:25]   # every 4th -> 25

# --- select 25 perturbation items, 5 per operator ---
OPS = ["entity_swap", "numeric_drift", "hedge_insertion", "distractor_parroting", "citation_relocation"]
by_op = {o: [] for o in OPS}
for sid in sorted(sp, key=lambda k: int(k.split("-")[1])):
    o = op_of.get(sid)
    if o in by_op:
        by_op[o].append(sid)
sel_sp = []
for o in OPS:
    sel_sp.extend(by_op[o][:5])

print(f"selected {len(sel_pn)} paraphrase + {len(sel_sp)} perturbation items")
print("perturbation per-operator:", {o: by_op[o][:5] for o in OPS})

# --- Task A packet (paraphrase equivalence) ---
taskA = ["# Second Annotator — Task A: Paraphrase Equivalence (25 items)",
         "",
         "Read INSTRUCTIONS.md first. For each item, decide whether **Answer B** conveys the",
         "EXACT SAME factual content as **Answer A**. Write yes/no in taskA_labels.csv",
         "(column `equivalent_yes_no`). Do NOT open any answer key or the first annotator's labels.",
         "", "---", ""]
for sid in sel_pn:
    taskA.append(pn[sid]); taskA.append("\n---\n")
(OUT / "taskA_paraphrase_packet.md").write_text("\n".join(taskA))

# --- Task B packet (perturbation faithfulness) ---
taskB = ["# Second Annotator — Task B: Perturbation Faithfulness (25 items)",
         "",
         "Read INSTRUCTIONS.md first. For each item, decide whether the **Candidate Answer** is",
         "FAITHFUL or UNFAITHFUL to the retrieved passages (judge relative to the passages, not",
         "world knowledge). Write faithful/unfaithful in taskB_labels.csv (column `faithful_yes_no`).",
         "Read every cited passage [N] before deciding — do not skim.",
         "", "---", ""]
for sid in sel_sp:
    taskB.append(sp[sid]); taskB.append("\n---\n")
(OUT / "taskB_perturbation_packet.md").write_text("\n".join(taskB))

# --- blank label CSVs ---
(OUT / "taskA_labels.csv").write_text(
    "item_id,equivalent_yes_no,notes\n" + "\n".join(f"{i},," for i in sel_pn) + "\n")
(OUT / "taskB_labels.csv").write_text(
    "item_id,faithful_yes_no,notes\n" + "\n".join(f"{i},," for i in sel_sp) + "\n")

# --- instructions (self-contained rubric, blind) ---
instructions = """# Second Annotator — Instructions (read fully before labeling)

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
"""
(OUT / "INSTRUCTIONS.md").write_text(instructions)

print(f"\nWrote packet to {OUT}/:")
for f in sorted(OUT.iterdir()):
    print(" ", f.name)
