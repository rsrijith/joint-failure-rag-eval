"""Score the human validation labels against intended labels and judge verdicts.

Task A (paraphrase equivalence): for paraphrases the human judged meaning-
preserving (equivalent=yes -> intended faithful), how often does each judge
call them UNFAITHFUL? That over-rejection rate is the real finding (vs the
generator's equivalence claim, which is self-judged by Claude).

Task B (perturbation faithfulness): human faithful/unfaithful vs intended
(all adversarial = unfaithful) vs each judge -> human-vs-judge agreement,
the validity gate. Broken down by operator.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HEADLINE = ["claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet",
            "hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large",
            "ragas_style_sonnet"]


def load_corrected_verdicts():
    """(seed_id, operator) -> {judge: verdict}, from the CORRECTED verdict files,
    so scoring uses current data (incl. fixed AlignScore), not the stale answer key."""
    out = defaultdict(dict)
    for base in ("results/preview_pilot/verdicts.jsonl",
                 "results/citation_relocation_pilot/verdicts.jsonl"):
        for line in Path(base).open():
            r = json.loads(line)
            if r["judge"] in HEADLINE and not r.get("metadata", {}).get("error"):
                out[(r["seed_id"], r["operator"])][r["judge"]] = r["verdict"]
    return out


def read_labels(path, value_col):
    """Robust read: skip a stray title line if present."""
    lines = Path(path).read_text().splitlines()
    # find the header line (the one containing 'item_id')
    start = next(i for i, l in enumerate(lines) if l.startswith("item_id"))
    reader = csv.DictReader(lines[start:])
    out = {}
    for r in reader:
        iid = (r.get("item_id") or "").strip()
        if not iid:
            continue
        out[iid] = {
            "label": (r.get(value_col) or "").strip().lower(),
            "notes": (r.get("notes") or "").strip(),
        }
    return out


def norm_yesno(v):
    if v.startswith("y"):
        return "yes"
    if v.startswith("n"):
        return "no"
    return v  # unexpected


def main():
    key = json.load(open("validation/_answer_key.json"))
    vmap = load_corrected_verdicts()

    # ---------- TASK A ----------
    a = read_labels("validation/paraphrase_equivalence_labels.csv", "equivalent_yes_no")
    a_norm = {k: norm_yesno(v["label"]) for k, v in a.items()}
    n = len(a_norm)
    yes = sum(1 for v in a_norm.values() if v == "yes")
    no = sum(1 for v in a_norm.values() if v == "no")
    other = n - yes - no
    print("=" * 70)
    print("  TASK A — Paraphrase equivalence")
    print("=" * 70)
    print(f"  {n} labeled: equivalent(yes)={yes}  not-equivalent(no)={no}  other={other}")
    print(f"  Human says {100*yes/n:.0f}% of paraphrases are meaning-preserving.")
    print(f"  (Generator/rule passed all of them as 'equivalent' — Claude self-judged.)")
    print()
    # For human-equivalent paraphrases, each judge's UNFAITHFUL rate = over-rejection
    print("  Over-rejection on HUMAN-equivalent paraphrases (judge says UNFAITHFUL")
    print("  on a paraphrase the human confirmed is meaning-preserving):")
    eq_items = [k for k, v in a_norm.items() if v == "yes"]
    rej = defaultdict(lambda: [0, 0])  # judge -> [unfaithful, total]
    for iid in eq_items:
        verds = vmap.get((key[iid]["seed_id"], key[iid]["operator"]), {})
        for j in HEADLINE:
            if j in verds:
                rej[j][1] += 1
                if verds[j] == "unfaithful":
                    rej[j][0] += 1
    for j in HEADLINE:
        u, t = rej[j]
        if t:
            print(f"    {j:<26} {u:>3}/{t:<3} = {100*u/t:>4.0f}% over-reject")
    if no:
        print(f"\n  The {no} 'not-equivalent' items (paraphrase drifted) — human notes:")
        for k, v in a.items():
            if norm_yesno(v["label"]) == "no":
                print(f"    {k}: {v['notes'][:100]}")

    # ---------- TASK B ----------
    b = read_labels("validation/perturbation_spotcheck_labels.csv", "faithful_yes_no")
    b_norm = {k: norm_yesno(v["label"]) for k, v in b.items()}
    nb = len(b_norm)
    print("\n" + "=" * 70)
    print("  TASK B — Perturbation faithfulness spot-check")
    print("=" * 70)
    faith = sum(1 for v in b_norm.values() if v == "yes")
    unfaith = sum(1 for v in b_norm.values() if v == "no")
    print(f"  {nb} labeled: human faithful={faith}  unfaithful={unfaith}")
    # human vs intended (all adversarial intended = unfaithful)
    agree_intended = sum(1 for iid, v in b_norm.items()
                         if (v == "no") == (key[iid]["intended_label"] == "unfaithful"))
    print(f"  Human vs intended label: {agree_intended}/{nb} = {100*agree_intended/nb:.0f}% agree")
    print(f"    (intended = unfaithful for all; human-unfaithful rate validates the operators)")
    # by operator
    by_op = defaultdict(lambda: [0, 0])  # op -> [human_unfaithful, total]
    for iid, v in b_norm.items():
        op = key[iid]["operator"]
        by_op[op][1] += 1
        if v == "no":
            by_op[op][0] += 1
    print("\n  Human-unfaithful rate by operator (higher = operator label is valid):")
    for op in sorted(by_op):
        u, t = by_op[op]
        print(f"    {op:<24} {u}/{t} judged unfaithful")
    # human vs each judge agreement
    print("\n  Human vs each judge agreement (on the 50 spot-check cells):")
    for j in HEADLINE:
        agree = tot = 0
        for iid, v in b_norm.items():
            verds = vmap.get((key[iid]["seed_id"], key[iid]["operator"]), {})
            if j in verds:
                tot += 1
                judge_unfaithful = verds[j] == "unfaithful"
                human_unfaithful = v == "no"
                if judge_unfaithful == human_unfaithful:
                    agree += 1
        if tot:
            print(f"    {j:<26} {agree:>3}/{tot:<3} = {100*agree/tot:>4.0f}% agree with human")
    if faith:
        print(f"\n  The {faith} items human judged FAITHFUL (disagree with intended) — notes:")
        for k, v in b.items():
            if norm_yesno(v["label"]) == "yes":
                print(f"    {k} ({key[k]['operator']}): {v['notes'][:100]}")


if __name__ == "__main__":
    main()
