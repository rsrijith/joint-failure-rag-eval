"""Combined 6-operator analysis (5 content-edit operators + citation_relocation).

Reads from BOTH:
    results/preview_pilot/{verdicts,perturbations}.jsonl   (5 operators)
    results/citation_relocation_pilot/{verdicts,perturbations}.jsonl

and produces unified per-judge FNR, joint-failure rate, pairwise Cohen's
kappa, and attack universality, treating citation_relocation as a 6th
operator stratum.

The two pilots use different clean baselines:
    - preview_pilot: operator="clean" = unmodified gold answer
    - citation_relocation_pilot: operator="clean_cited" = Claude-annotated gold
both of which are excluded from the perturbation analysis.

Run from repo root:
    python scripts/analyze_combined.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OUT_DIRS = [
    Path("results/preview_pilot"),
    Path("results/citation_relocation_pilot"),
]
CLEAN_OPERATORS = {"clean", "clean_cited"}  # not perturbations


def _cohens_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    cats = sorted(set(a) | set(b))
    if len(cats) <= 1:
        return 1.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for c in cats:
        p_a = sum(1 for x in a if x == c) / n
        p_b = sum(1 for x in b if x == c) / n
        pe += p_a * p_b
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _load_all() -> tuple[list[dict], list[dict]]:
    verdicts: list[dict] = []
    perturbations: list[dict] = []
    for d in OUT_DIRS:
        v_path = d / "verdicts.jsonl"
        p_path = d / "perturbations.jsonl"
        if v_path.exists():
            verdicts.extend(json.loads(line) for line in v_path.open())
        if p_path.exists():
            perturbations.extend(json.loads(line) for line in p_path.open())
    return verdicts, perturbations


def main() -> None:
    verdicts, perturbations = _load_all()
    if not verdicts:
        raise SystemExit(f"No verdicts found across {OUT_DIRS}")

    all_judges = sorted({v["judge"] for v in verdicts})
    judges = sorted({
        v["judge"] for v in verdicts if v["operator"] not in CLEAN_OPERATORS
    })
    operators = sorted({
        v["operator"] for v in verdicts if v["operator"] not in CLEAN_OPERATORS
    })
    excluded = set(all_judges) - set(judges)

    print(f"Loaded {len(verdicts)} verdicts, {len(perturbations)} perturbations across {len(OUT_DIRS)} pilots.")
    print(f"Judges with perturbed-answer verdicts ({len(judges)}): {judges}")
    if excluded:
        print(f"Excluded judges (no perturbed verdicts): {sorted(excluded)}")
    print(f"Operators ({len(operators)}): {operators}")
    print()
    print("Note: paraphrase_null is a NEGATIVE CONTROL — perturbed answers are")
    print("semantically equivalent to gold and should be judged faithful. A high")
    print("'FNR' on paraphrase_null is the CORRECT outcome.")
    print()
    print("Note: citation_relocation has a different failure semantics — claims")
    print("remain in context; only citation attribution is broken. Judges that")
    print("verify claim-in-context without checking per-citation attribution")
    print("are EXPECTED to call these perturbations faithful, exposing a shared")
    print("blind spot across the ensemble.")
    print()

    # ----- Rule-pass per operator -----
    print("=" * 78)
    print("Perturbation suite: rule-pass rate per operator")
    print("=" * 78)
    op_pass: dict[str, list[bool]] = defaultdict(list)
    for p in perturbations:
        op_pass[p["operator"]].append(p["rule_passed"])
    for op in operators:
        n = len(op_pass[op])
        n_pass = sum(op_pass[op])
        pct = 100 * n_pass / max(1, n)
        print(f"  {op:22s} {n_pass:3d}/{n:3d} passed ({pct:5.1f}%)")
    print()

    # Build valid-perturbed set + verdict map
    valid_perturbed = {(p["seed_id"], p["operator"]) for p in perturbations if p["rule_passed"]}
    verdict_map: dict[tuple, str] = {}
    for v in verdicts:
        verdict_map[(v["seed_id"], v["operator"], v["judge"])] = v["verdict"]

    # ----- Per-judge FNR per operator -----
    print("=" * 78)
    print("Per-judge false-negative rate (marginals)")
    print("=" * 78)
    print("FNR = fraction of perturbations the judge called 'faithful'.\n")
    print(f"{'operator':22s} | " + " | ".join(f"{j:14s}" for j in judges))
    print("-" * (22 + 3 + (14 + 3) * len(judges)))
    for op in operators:
        cells = []
        for j in judges:
            vfor = [
                verdict_map.get((sid, op, j))
                for (sid, oo) in valid_perturbed
                if oo == op and (sid, op, j) in verdict_map
            ]
            n_fnr = sum(1 for v in vfor if v == "faithful")
            n_total = len(vfor)
            pct = 100 * n_fnr / max(1, n_total)
            cells.append(f"{n_fnr:3d}/{n_total:3d}({pct:3.0f}%)")
        print(f"{op:22s} | " + " | ".join(f"{c:14s}" for c in cells))
    print()

    # ----- Joint failure + attack universality per operator -----
    print("=" * 78)
    print("Joint-failure rate per operator (ALL judges call perturbed 'faithful')")
    print("=" * 78)
    print(f"Ensemble size: {len(judges)} judges.\n")
    for op in operators:
        n_joint_fail = 0
        n_evaluated = 0
        for sid, oo in valid_perturbed:
            if oo != op:
                continue
            row = [verdict_map.get((sid, op, j)) for j in judges]
            if any(v is None for v in row):
                continue
            n_evaluated += 1
            if all(v == "faithful" for v in row):
                n_joint_fail += 1
        rate = 100 * n_joint_fail / max(1, n_evaluated)

        marginals = []
        for j in judges:
            n_fnr = sum(
                1 for sid, oo in valid_perturbed
                if oo == op and verdict_map.get((sid, op, j)) == "faithful"
            )
            n_tot = sum(1 for sid, oo in valid_perturbed if oo == op)
            marginals.append(n_fnr / max(1, n_tot))
        independence = 1.0
        for m in marginals:
            independence *= m
        universality = (n_joint_fail / max(1, n_evaluated)) - independence

        print(f"  {op:22s} joint failure {n_joint_fail:3d}/{n_evaluated:3d} = {rate:5.1f}%   "
              f"baseline={100*independence:5.2f}%   universality={100*universality:+.2f}pp")
    print()

    # ----- Pairwise Cohen's kappa across judges -----
    print("=" * 78)
    print("Pairwise Cohen's kappa across judges (perturbed answers, all operators pooled)")
    print("=" * 78)
    print("Higher kappa = judges agree more (positive correlation).\n")
    per_judge_v: dict[str, list[str]] = {j: [] for j in judges}
    aligned = 0
    for sid, op in valid_perturbed:
        row = {j: verdict_map.get((sid, op, j)) for j in judges}
        if any(v is None for v in row.values()):
            continue
        aligned += 1
        for j, v in row.items():
            per_judge_v[j].append(v)
    print(f"Aligned on {aligned} (seed, operator) cells.\n")
    print(f"{'':18s} | " + " | ".join(f"{j:14s}" for j in judges))
    print("-" * (18 + 3 + (14 + 3) * len(judges)))
    for j1 in judges:
        cells = []
        for j2 in judges:
            if j1 == j2:
                cells.append(f"{'-':>14s}")
            else:
                k = _cohens_kappa(per_judge_v[j1], per_judge_v[j2])
                cells.append(f"{k:+.3f}")
        print(f"{j1:18s} | " + " | ".join(f"{c:14s}" for c in cells))
    print()

    # ----- Operator-stratified joint-failure summary -----
    print("=" * 78)
    print("Operator-stratified summary (for paper headline numbers)")
    print("=" * 78)
    adv_ops = [op for op in operators if op != "paraphrase_null"]
    n_meets = 0
    for op in adv_ops:
        n_joint_fail = sum(
            1 for sid, oo in valid_perturbed
            if oo == op and all(
                verdict_map.get((sid, op, j)) == "faithful" for j in judges
            ) and all(
                verdict_map.get((sid, op, j)) is not None for j in judges
            )
        )
        n_eval = sum(
            1 for sid, oo in valid_perturbed
            if oo == op and all(verdict_map.get((sid, op, j)) is not None for j in judges)
        )
        rate = 100 * n_joint_fail / max(1, n_eval)
        meets = rate >= 25.0
        if meets:
            n_meets += 1
        marker = "  PASS (>=25%)" if meets else ""
        print(f"  {op:22s} joint-failure {rate:5.1f}%{marker}")
    print(f"\n  {n_meets}/{len(adv_ops)} adversarial operators clear the 25% joint-failure threshold.")


if __name__ == "__main__":
    main()
