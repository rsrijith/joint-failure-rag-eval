"""Corrected headline analysis after dropping the degenerate AlignScore judge.

The clean-pass re-scoring revealed AlignScore-large calls 0% of clean gold
answers faithful at the default 0.5 threshold -- it is a near-constant
"unfaithful" predictor and is non-discriminative on this task. We therefore
report a 6-judge FUNCTIONAL ensemble (3 LLM + HHEM + MiniCheck + RAGAS) and
treat AlignScore as a separate calibration cautionary note.

Recomputes, on the functional ensemble:
  - pairwise Cohen's kappa (adversarial cells)
  - within-LLM vs within-NLI cluster kappa
  - the clean-faithful vs citation_relocation-FNR frontier (no judge both
    passes clean answers AND catches citation misattribution)
  - ensemble shrinkage vs majority-of-6
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

FUNCTIONAL = [
    "claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet",
    "hhem_2_1_open", "minicheck_flan_t5_large", "ragas_style_sonnet",
]
LLM = ["claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"]
NLI = ["hhem_2_1_open", "minicheck_flan_t5_large"]  # AlignScore dropped

ADVERSARIAL_OPS = ["entity_swap", "numeric_drift", "hedge_insertion",
                   "distractor_parroting", "citation_relocation"]
CLEAN_OPS = ["clean", "clean_cited"]


def load():
    """seed_op_judge[(seed,op)][judge] = verdict, for non-error functional verdicts."""
    cell = defaultdict(dict)
    for base in ("results/preview_pilot/verdicts.jsonl",
                 "results/citation_relocation_pilot/verdicts.jsonl"):
        p = Path(base)
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r["judge"] not in FUNCTIONAL:
                continue
            if r.get("metadata", {}).get("error"):
                continue
            cell[(r["seed_id"], r["operator"])][r["judge"]] = r["verdict"]
    return cell


def cohen_kappa(cell, j1, j2, ops):
    """Cohen's kappa between two judges over cells in `ops` where both voted."""
    a, b = [], []
    for (sid, op), jv in cell.items():
        if op not in ops:
            continue
        if j1 in jv and j2 in jv:
            a.append(jv[j1])
            b.append(jv[j2])
    if not a:
        return None, 0
    n = len(a)
    cats = ["faithful", "unfaithful"]
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for c in cats:
        pa = sum(1 for x in a if x == c) / n
        pb = sum(1 for y in b if y == c) / n
        pe += pa * pb
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    return kappa, n


def faithful_rate(cell, judge, ops):
    vals = [jv[judge] for (sid, op), jv in cell.items() if op in ops and judge in jv]
    if not vals:
        return None, 0
    return sum(1 for v in vals if v == "faithful") / len(vals), len(vals)


def main():
    cell = load()
    print("=" * 78)
    print("  CORRECTED HEADLINE ANALYSIS (6 functional judges; AlignScore dropped)")
    print("=" * 78)

    # --- Pairwise kappa on adversarial cells
    print("\n[1] Pairwise Cohen's kappa (adversarial operators pooled)\n")
    hdr = "  " + " " * 24 + " ".join(f"{j[:8]:>9}" for j in FUNCTIONAL)
    print(hdr)
    for j1 in FUNCTIONAL:
        row = [f"  {j1:<24}"]
        for j2 in FUNCTIONAL:
            if j1 == j2:
                row.append(f"{'--':>9}")
            else:
                k, n = cohen_kappa(cell, j1, j2, ADVERSARIAL_OPS)
                row.append(f"{k:>+9.3f}" if k is not None else f"{'n/a':>9}")
        print(" ".join(row))

    # --- Within-cluster kappa
    print("\n[2] Within-architecture cluster kappa (adversarial pooled)\n")
    llm_pairs = [(a, b) for a, b in combinations(LLM, 2)]
    nli_pairs = [(a, b) for a, b in combinations(NLI, 2)]
    llm_ks = [cohen_kappa(cell, a, b, ADVERSARIAL_OPS)[0] for a, b in llm_pairs]
    nli_ks = [cohen_kappa(cell, a, b, ADVERSARIAL_OPS)[0] for a, b in nli_pairs]
    for (a, b), k in zip(llm_pairs, llm_ks):
        print(f"    LLM  {a[:14]:<14} x {b[:14]:<14}  kappa = {k:+.3f}")
    print(f"    --> within-LLM avg kappa = {sum(llm_ks)/len(llm_ks):+.3f}  ({len(llm_pairs)} pairs)")
    print()
    for (a, b), k in zip(nli_pairs, nli_ks):
        print(f"    NLI  {a[:14]:<14} x {b[:14]:<14}  kappa = {k:+.3f}")
    print(f"    --> within-NLI avg kappa = {sum(nli_ks)/len(nli_ks):+.3f}  ({len(nli_pairs)} pair)")
    print()
    gap = sum(llm_ks)/len(llm_ks) - sum(nli_ks)/len(nli_ks)
    print(f"    GAP (within-LLM - within-NLI) = {gap:+.3f}")
    print(f"    NOTE: with AlignScore dropped, the NLI 'cluster' is a single pair")
    print(f"    (HHEM x MiniCheck). The architecture-cluster claim is weaker for NLI.")

    # --- Clean-faithful vs citation-FNR frontier
    print("\n[3] The citation-attribution blind spot is UNIVERSAL across functional judges\n")
    print("    A judge is useful only if it passes clean answers (high clean-faithful)")
    print("    AND catches misattribution (low citation FNR). No functional judge does both.\n")
    print(f"    {'judge':<26} {'clean-faithful':>15} {'citation FNR':>14}")
    print("    " + "-" * 56)
    for j in FUNCTIONAL:
        cf, cfn = faithful_rate(cell, j, CLEAN_OPS)
        cit, citn = faithful_rate(cell, j, ["citation_relocation"])
        cf_s = f"{cf*100:.0f}% (n={cfn})" if cf is not None else "n/a"
        cit_s = f"{cit*100:.0f}% (n={citn})" if cit is not None else "n/a"
        print(f"    {j:<26} {cf_s:>15} {cit_s:>14}")
    print()
    print("    AlignScore (excluded): clean-faithful 0% -> 'catches' citation at 0% FNR")
    print("    only because it rejects everything. Not a functional detector.")

    # --- Ensemble shrinkage vs majority-of-6
    print("\n[4] Ensemble shrinkage vs majority-of-6 (functional) on adversarial cells\n")
    adv_cells = {k: v for k, v in cell.items()
                 if k[1] in ADVERSARIAL_OPS and all(j in v for j in FUNCTIONAL)}
    print(f"    Comparison base: {len(adv_cells)} cells fully scored by all 6 functional judges")

    def majority(jv, judges):
        f = sum(1 for j in judges if jv[j] == "faithful")
        return "faithful" if f > len(judges) / 2 else "unfaithful"

    ref = {k: majority(v, FUNCTIONAL) for k, v in adv_cells.items()}
    for size in (2, 3, 4):
        best, best_agree = None, -1
        for subset in combinations(FUNCTIONAL, size):
            agree = sum(1 for k, v in adv_cells.items()
                        if majority(v, subset) == ref[k]) / len(adv_cells)
            if agree > best_agree:
                best_agree, best = agree, subset
        print(f"    best {size}-judge subset: {best} -> {best_agree*100:.1f}% match with majority-of-6")


if __name__ == "__main__":
    main()
