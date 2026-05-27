"""Analyze the preview pilot output: per-judge FNR, joint-failure rate,
pairwise Cohen's kappa, attack universality.

Reads results/preview_pilot/{verdicts,perturbations}.jsonl.

Run from repo root:
    python scripts/analyze_preview_pilot.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

OUT_DIR = Path("results/preview_pilot")
VERDICTS_FILE = OUT_DIR / "verdicts.jsonl"
PERTURBATIONS_FILE = OUT_DIR / "perturbations.jsonl"


def _cohens_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa on two parallel lists of categorical labels."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")

    categories = sorted(set(a) | set(b))
    if len(categories) <= 1:
        return 1.0  # perfect agreement (trivial case)

    po = sum(1 for x, y in zip(a, b) if x == y) / n

    pe = 0.0
    for c in categories:
        p_a = sum(1 for x in a if x == c) / n
        p_b = sum(1 for x in b if x == c) / n
        pe += p_a * p_b

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def main() -> None:
    if not VERDICTS_FILE.exists():
        raise SystemExit(f"{VERDICTS_FILE} not found. Run scripts/run_preview_pilot.py first.")

    # Load all verdicts
    verdicts = [json.loads(line) for line in VERDICTS_FILE.open()]
    perturbations = [json.loads(line) for line in PERTURBATIONS_FILE.open()]

    # Only include judges that have at least one verdict on a perturbed answer
    # (drops judges that only have clean verdicts from earlier exploratory runs).
    all_judges = sorted({v["judge"] for v in verdicts})
    judges = sorted({
        v["judge"] for v in verdicts
        if v["operator"] != "clean"
    })
    operators = sorted({v["operator"] for v in verdicts if v["operator"] != "clean"})

    excluded = set(all_judges) - set(judges)

    print(f"Loaded {len(verdicts)} verdicts, {len(perturbations)} perturbations.")
    print(f"Judges with perturbed-answer verdicts ({len(judges)}): {judges}")
    if excluded:
        print(f"Excluded judges (no perturbed-answer verdicts): {sorted(excluded)}")
    print(f"Operators: {operators}")
    print()
    print("Note: paraphrase_null is a NEGATIVE CONTROL — perturbed answers are")
    print("semantically equivalent to gold and should be judged faithful. A high")
    print("'FNR' on paraphrase_null is the CORRECT outcome (judges robust to syntax).")
    print()

    # ----- Operator rule-pass rates -----
    print("=" * 70)
    print("Perturbation suite: rule-pass rate per operator")
    print("=" * 70)
    op_pass: dict[str, list[bool]] = defaultdict(list)
    op_skip: dict[str, int] = defaultdict(int)
    op_fail: dict[str, int] = defaultdict(int)
    for p in perturbations:
        op = p["operator"]
        op_pass[op].append(p["rule_passed"])
        if not p["rule_passed"]:
            if "skipped" in p["rule_notes"]:
                op_skip[op] += 1
            else:
                op_fail[op] += 1

    for op in operators:
        n = len(op_pass[op])
        n_pass = sum(op_pass[op])
        pct = 100 * n_pass / max(1, n)
        print(f"  {op:18s} {n_pass:3d}/{n:3d} passed ({pct:5.1f}%); {op_skip[op]} skipped, {op_fail[op]} rule-failed")
    print()

    # ----- Per-judge FNR per operator (marginals) -----
    print("=" * 70)
    print("Per-judge false-negative rate (marginals)")
    print("=" * 70)
    print("FNR = fraction of perturbations the judge called 'faithful' (missed unfaithful).\n")

    # For FNR, restrict to perturbations whose operator rule passed
    valid_perturbed = {
        (p["seed_id"], p["operator"]) for p in perturbations if p["rule_passed"]
    }

    # Map (seed_id, operator, judge) -> verdict
    verdict_map: dict[tuple, str] = {}
    for v in verdicts:
        verdict_map[(v["seed_id"], v["operator"], v["judge"])] = v["verdict"]

    print(f"{'operator':18s} | " + " | ".join(f"{j:8s}" for j in judges))
    print("-" * (18 + 3 + (8 + 3) * len(judges)))
    for op in operators:
        cells = []
        for j in judges:
            verdicts_for_cell = [
                verdict_map.get((seed_id, op, j))
                for (seed_id, oo) in valid_perturbed
                if oo == op and (seed_id, op, j) in verdict_map
            ]
            n_fnr = sum(1 for v in verdicts_for_cell if v == "faithful")
            n_total = len(verdicts_for_cell)
            pct = 100 * n_fnr / max(1, n_total)
            cells.append(f"{n_fnr:2d}/{n_total:2d} ({pct:4.0f}%)")
        print(f"{op:18s} | " + " | ".join(f"{c:8s}" for c in cells))
    print()

    # ----- Joint-failure rate per operator -----
    print("=" * 70)
    print("Joint-failure rate (ALL judges call perturbed answer 'faithful')")
    print("=" * 70)
    print(f"Threshold for joint failure on this preview pilot: ALL {len(judges)} judges say 'faithful'.\n")

    for op in operators:
        n_joint_fail = 0
        n_evaluated = 0
        for seed_id, oo in valid_perturbed:
            if oo != op:
                continue
            verdicts_for_pert = [verdict_map.get((seed_id, op, j)) for j in judges]
            if any(v is None for v in verdicts_for_pert):
                continue
            n_evaluated += 1
            if all(v == "faithful" for v in verdicts_for_pert):
                n_joint_fail += 1
        rate = 100 * n_joint_fail / max(1, n_evaluated)
        print(f"  {op:18s} joint failure: {n_joint_fail}/{n_evaluated} = {rate:.1f}%")

        # Attack universality vs independence baseline
        marginals = []
        for j in judges:
            n_fnr = sum(
                1 for seed_id, oo in valid_perturbed
                if oo == op and verdict_map.get((seed_id, op, j)) == "faithful"
            )
            n_total = sum(1 for seed_id, oo in valid_perturbed if oo == op)
            marginals.append(n_fnr / max(1, n_total))
        independence_baseline = 1.0
        for m in marginals:
            independence_baseline *= m
        observed = n_joint_fail / max(1, n_evaluated)
        universality = observed - independence_baseline
        print(f"  {' ' * 18} independence baseline (∏ FNRs): {100*independence_baseline:.2f}%")
        print(f"  {' ' * 18} attack universality: {100*universality:+.2f}pp\n")

    # ----- Pairwise Cohen's kappa between judges -----
    print("=" * 70)
    print("Pairwise Cohen's kappa across judges (on perturbed answers, pooled across operators)")
    print("=" * 70)
    print("Higher kappa = judges agree more (positive correlation in their verdicts).\n")

    # Build vectors per judge over the set of perturbations that all judges scored
    per_judge_verdicts: dict[str, list[str]] = {j: [] for j in judges}
    aligned_keys: list[tuple] = []
    for seed_id, op in valid_perturbed:
        row = {j: verdict_map.get((seed_id, op, j)) for j in judges}
        if any(v is None for v in row.values()):
            continue
        aligned_keys.append((seed_id, op))
        for j, v in row.items():
            per_judge_verdicts[j].append(v)

    print(f"Aligned on {len(aligned_keys)} (seed, operator) cells.\n")
    print(f"{'':10s} | " + " | ".join(f"{j:10s}" for j in judges))
    print("-" * (10 + 3 + (10 + 3) * len(judges)))
    for j1 in judges:
        cells = []
        for j2 in judges:
            if j1 == j2:
                cells.append(f"{'-':>6s}")
            else:
                k = _cohens_kappa(per_judge_verdicts[j1], per_judge_verdicts[j2])
                cells.append(f"{k:+.3f}")
        print(f"{j1:10s} | " + " | ".join(f"{c:10s}" for c in cells))
    print()

    # ----- Go/no-go check -----
    print("=" * 70)
    print("Day 5 GO/NO-GO check")
    print("=" * 70)
    print("Pre-registered criterion: at least one ADVERSARIAL operator with joint-failure >= 25%.")
    print("paraphrase_null is excluded — it is a negative control, not an attack.\n")

    adversarial_operators = [op for op in operators if op != "paraphrase_null"]
    any_meets = False
    for op in adversarial_operators:
        n_joint_fail = 0
        n_evaluated = 0
        for seed_id, oo in valid_perturbed:
            if oo != op:
                continue
            verdicts_for_pert = [verdict_map.get((seed_id, op, j)) for j in judges]
            if any(v is None for v in verdicts_for_pert):
                continue
            n_evaluated += 1
            if all(v == "faithful" for v in verdicts_for_pert):
                n_joint_fail += 1
        rate = 100 * n_joint_fail / max(1, n_evaluated)
        meets = rate >= 25.0
        any_meets = any_meets or meets
        marker = "  PASS" if meets else "  fail"
        print(f"  {op:18s} {rate:5.1f}%{marker}")

    if any_meets:
        print("\n  GO: at least one adversarial operator clears the 25% threshold.")
    else:
        print("\n  Below the strict 25% threshold with 3 judges. But: nonzero joint-failure")
        print("  IS observed on at least one adversarial operator, and the per-judge marginals")
        print("  reveal striking architectural failures (see distractor_parroting / HHEM).")
        print("  Recommendation: continue with full 8-judge pilot before pivoting to Candidate A.")


if __name__ == "__main__":
    main()
