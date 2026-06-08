"""Score the second human annotator against the first annotator (human-human IAA).

Run AFTER validation/second_annotator/taskA_labels.csv and taskB_labels.csv are filled.
Computes, on the overlapping items:
  - Task A (paraphrase equivalence) human-human agreement + Cohen's kappa
  - Task B (perturbation faithfulness) human-human agreement + Cohen's kappa
  - Combined balanced-gold faithfulness kappa (Task A equivalent->faithful, Task B as-is)
This is the second-independent-human number the reviewers asked for, to sit alongside the
human-vs-Mistral kappa=0.891 already reported.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

VAL = Path("validation")
A2 = VAL / "second_annotator"


def read_csv(path, key, val):
    out = {}
    if not Path(path).exists():
        return out
    lines = Path(path).read_text().splitlines()
    start = next((i for i, l in enumerate(lines) if l.lower().lstrip("﻿").startswith("item_id")), 0)
    for row in csv.DictReader(io.StringIO("\n".join(lines[start:]))):
        rid = (row.get("item_id") or row.get("﻿item_id") or "").strip()
        v = (row.get(val) or "").strip().lower()
        if rid and v:
            out[rid] = v
    return out


def kappa(pairs):
    """pairs: list of (a,b) binary labels (strings)."""
    n = len(pairs)
    if n == 0:
        return float("nan"), 0, 0.0
    agree = sum(1 for a, b in pairs if a == b)
    po = agree / n
    labels = sorted({x for p in pairs for x in p})
    # expected by chance from marginals
    pe = 0.0
    for L in labels:
        pa = sum(1 for a, _ in pairs if a == L) / n
        pb = sum(1 for _, b in pairs if b == L) / n
        pe += pa * pb
    k = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return k, agree, n


def norm_faith(v):
    return "faithful" if v.startswith("f") or v == "yes" else "unfaithful"


# Task A: equivalence
a1 = read_csv(VAL / "paraphrase_equivalence_labels.csv", "item_id", "equivalent_yes_no")
a2 = read_csv(A2 / "taskA_labels.csv", "item_id", "equivalent_yes_no")
common_a = sorted(set(a1) & set(a2))
pairs_a = [(a1[i], a2[i]) for i in common_a]
ka, agA, nA = kappa(pairs_a)

# Task B: faithfulness
b1 = read_csv(VAL / "perturbation_spotcheck_labels.csv", "item_id", "faithful_yes_no")
b2 = read_csv(A2 / "taskB_labels.csv", "item_id", "faithful_yes_no")
common_b = sorted(set(b1) & set(b2))
pairs_b = [(norm_faith(b1[i]), norm_faith(b2[i])) for i in common_b]
kb, agB, nB = kappa(pairs_b)

# Combined balanced-gold faithfulness: paraphrase equivalent->faithful
def equiv_to_faith(v):
    return "faithful" if v.startswith("y") else "unfaithful"
pairs_combined = [(equiv_to_faith(a1[i]), equiv_to_faith(a2[i])) for i in common_a] + pairs_b
kc, agC, nC = kappa(pairs_combined)

print("=" * 60)
print("SECOND HUMAN ANNOTATOR vs FIRST ANNOTATOR (inter-annotator agreement)")
print("=" * 60)
if nA:
    print(f"Task A paraphrase equivalence : {agA}/{nA} agree ({100*agA/nA:.0f}%), kappa = {ka:.3f}")
else:
    print("Task A: second-annotator labels not found/filled yet.")
if nB:
    print(f"Task B perturbation faithfulness: {agB}/{nB} agree ({100*agB/nB:.0f}%), kappa = {kb:.3f}")
else:
    print("Task B: second-annotator labels not found/filled yet.")
if nC:
    print(f"Combined faithfulness gold (50) : {agC}/{nC} agree ({100*agC/nC:.0f}%), kappa = {kc:.3f}")
print("\nPaper line (fill from above): 'a second independent human annotator on a 50-item")
print("stratified subset agreed with the first at Cohen's kappa = <combined>, alongside the")
print("human-vs-model kappa = 0.891.'")
