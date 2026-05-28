"""Hypothesis validation v2: rigorous tests for H-A (LLM-judge redundancy)
and H-D (within-architecture diversity reversal), plus exploratory analyses
for additional angles.

H-A: "LLM-judge ensembling is redundant"
  Tests:
    A1. Within-LLM-cluster avg pairwise κ ≥ 0.5 (organization-diverse but
        verdict-correlated)
    A2. Marginal information gain from adding a 3rd LLM-judge to a 2-LLM
        ensemble: should be SMALL (i.e. 3rd judge rarely changes the majority)
    A3. Stability of high LLM-cluster κ across operators (not just pooled)

H-D: "Within-architecture diversity reversal"
  Tests:
    D1. Within-LLM κ > Within-NLI κ (the reversal itself)
    D2. The gap (within-LLM - within-NLI) ≥ 0.3 (large enough to matter)
    D3. Stable across operators (i.e. always the same direction, not a pooled artifact)

Plus exploratory:
  E1. Architecture × operator FNR matrix — do specific architectures have
      consistent blind spots for specific operators?
  E2. Ensemble-shrinkage analysis — what's the smallest ensemble that
      captures 90% of the full 7-judge verdict variance?
  E3. AlignScore standalone — what does it actually predict?

Run from repo root:
    python scripts/hypothesis_check_v2.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev

OUT_DIRS = [Path("results/preview_pilot"), Path("results/citation_relocation_pilot")]
CLEAN_OPS = {"clean", "clean_cited"}
EXCLUDED_JUDGES = {"glm_4_7_cerebras", "qwen3_235b_cerebras", "claude_opus_4_7"}

LLM_CLUSTER = {"claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"}
NLI_CLUSTER = {"hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large"}
DECOMP_CLUSTER = {"ragas_style_sonnet"}


def _cohens_kappa(a, b):
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


def _load():
    verdicts, perts = [], []
    for d in OUT_DIRS:
        v = d / "verdicts.jsonl"
        p = d / "perturbations.jsonl"
        if v.exists():
            verdicts.extend(json.loads(line) for line in v.open())
        if p.exists():
            perts.extend(json.loads(line) for line in p.open())
    return verdicts, perts


def main():
    verdicts, perts = _load()
    judges = sorted({
        v["judge"] for v in verdicts
        if v["operator"] not in CLEAN_OPS and v["judge"] not in EXCLUDED_JUDGES
    })
    operators = sorted({v["operator"] for v in verdicts if v["operator"] not in CLEAN_OPS})
    adversarial_ops = [op for op in operators if op != "paraphrase_null"]

    valid_perturbed = {(p["seed_id"], p["operator"]) for p in perts if p["rule_passed"]}
    verdict_map = {(v["seed_id"], v["operator"], v["judge"]): v["verdict"] for v in verdicts}

    # Filter to fully-scored cells (all 7 judges have a verdict)
    full_cells = []
    for (sid, op) in valid_perturbed:
        if all((sid, op, j) in verdict_map for j in judges):
            full_cells.append((sid, op))
    print(f"Fully-scored adversarial cells: {sum(1 for (s,o) in full_cells if o!='paraphrase_null')}")
    print(f"Total fully-scored cells (incl negative control): {len(full_cells)}")
    print()

    # ================================================================
    # H-A: LLM-JUDGE REDUNDANCY
    # ================================================================
    print("=" * 78)
    print("H-A: LLM-judge ensembling is redundant")
    print("=" * 78)

    # A1: pooled within-LLM κ
    print("\n  A1: pairwise κ within LLM cluster (pooled across operators)")
    llm_judges = sorted(LLM_CLUSTER & set(judges))
    per_judge_v = {j: [] for j in llm_judges}
    for sid, op in full_cells:
        if op == "paraphrase_null":
            continue
        for j in llm_judges:
            per_judge_v[j].append(verdict_map[(sid, op, j)])

    llm_pairs = list(combinations(llm_judges, 2))
    llm_kappas = []
    for j1, j2 in llm_pairs:
        k = _cohens_kappa(per_judge_v[j1], per_judge_v[j2])
        llm_kappas.append(k)
        print(f"    {j1} vs {j2}: κ = {k:+.3f}")
    llm_kappa_avg = mean(llm_kappas) if llm_kappas else 0.0
    print(f"  Within-LLM avg κ = {llm_kappa_avg:+.3f}  → A1 {'PASS' if llm_kappa_avg >= 0.5 else 'FAIL'} (threshold 0.5)")

    # A2: marginal information gain — how often does the 3rd LLM-judge
    # flip the LLM-cluster majority?
    print("\n  A2: marginal info gain from a 3rd LLM-judge")
    if len(llm_judges) >= 3:
        n_majority_flipped = 0
        n_total = 0
        for sid, op in full_cells:
            if op == "paraphrase_null":
                continue
            n_total += 1
            verdicts_3 = [verdict_map[(sid, op, j)] for j in llm_judges]
            # Does removing any one judge flip the majority?
            full_majority = "faithful" if verdicts_3.count("faithful") >= 2 else "unfaithful"
            flipped = False
            for i in range(3):
                subset = verdicts_3[:i] + verdicts_3[i+1:]  # 2 judges
                sub_maj = "faithful" if subset.count("faithful") == 2 else (
                    "unfaithful" if subset.count("unfaithful") == 2 else "tied"
                )
                if sub_maj == "tied":
                    continue
                if sub_maj != full_majority:
                    flipped = True
                    break
            if flipped:
                n_majority_flipped += 1
        flip_rate = n_majority_flipped / max(1, n_total)
        print(f"    3rd-judge flips LLM-majority verdict on {n_majority_flipped}/{n_total} = {100*flip_rate:.1f}% of cells")
        a2_pass = flip_rate < 0.10
        print(f"  A2 {'PASS' if a2_pass else 'FAIL'} (threshold <10% — 3rd judge adds <10% information)")

    # A3: per-operator stability of within-LLM κ
    print("\n  A3: per-operator within-LLM κ stability")
    per_op_llm_kappa = {}
    for op in adversarial_ops:
        op_cells = [(s, o) for (s, o) in full_cells if o == op]
        if len(op_cells) < 10:
            continue
        op_v = {j: [] for j in llm_judges}
        for sid, oo in op_cells:
            for j in llm_judges:
                op_v[j].append(verdict_map[(sid, oo, j)])
        op_ks = [_cohens_kappa(op_v[j1], op_v[j2]) for j1, j2 in llm_pairs]
        per_op_llm_kappa[op] = mean(op_ks)
        print(f"    {op:22s} within-LLM avg κ = {per_op_llm_kappa[op]:+.3f}  (n={len(op_cells)})")
    if per_op_llm_kappa:
        op_kappas = list(per_op_llm_kappa.values())
        print(f"  Range: [{min(op_kappas):+.3f}, {max(op_kappas):+.3f}]")
        print(f"  All > 0.4: {'YES' if all(k >= 0.4 for k in op_kappas) else 'NO'}")
        a3_pass = all(k >= 0.4 for k in op_kappas) and (max(op_kappas) - min(op_kappas) < 0.3)
        print(f"  A3 {'PASS' if a3_pass else 'FAIL'} (stable high κ across operators)")

    # ================================================================
    # H-D: WITHIN-ARCHITECTURE DIVERSITY REVERSAL
    # ================================================================
    print()
    print("=" * 78)
    print("H-D: Within-architecture diversity reversal (LLM clusters but NLI doesn't)")
    print("=" * 78)

    # D1 + D2: pooled within-NLI κ + gap
    nli_judges = sorted(NLI_CLUSTER & set(judges))
    per_nli_v = {j: [] for j in nli_judges}
    for sid, op in full_cells:
        if op == "paraphrase_null":
            continue
        for j in nli_judges:
            per_nli_v[j].append(verdict_map[(sid, op, j)])
    nli_pairs = list(combinations(nli_judges, 2))
    print("\n  D1: pairwise κ within NLI cluster (pooled)")
    nli_kappas = []
    for j1, j2 in nli_pairs:
        k = _cohens_kappa(per_nli_v[j1], per_nli_v[j2])
        nli_kappas.append(k)
        print(f"    {j1} vs {j2}: κ = {k:+.3f}")
    nli_kappa_avg = mean(nli_kappas) if nli_kappas else 0.0
    print(f"  Within-NLI avg κ = {nli_kappa_avg:+.3f}")

    gap = llm_kappa_avg - nli_kappa_avg
    print(f"\n  D1+D2: within-LLM ({llm_kappa_avg:+.3f}) - within-NLI ({nli_kappa_avg:+.3f}) = {gap:+.3f}")
    d12_pass = gap >= 0.30
    print(f"  D1+D2 {'PASS' if d12_pass else 'FAIL'} (threshold gap ≥ 0.30)")

    # D3: per-operator stability of the reversal
    print("\n  D3: per-operator stability — is the reversal stable?")
    n_ops_reversed = 0
    for op in adversarial_ops:
        op_cells = [(s, o) for (s, o) in full_cells if o == op]
        if len(op_cells) < 10:
            continue
        op_v = {j: [] for j in judges}
        for sid, oo in op_cells:
            for j in judges:
                op_v[j].append(verdict_map[(sid, oo, j)])
        op_llm = mean(_cohens_kappa(op_v[a], op_v[b]) for a, b in llm_pairs)
        op_nli = mean(_cohens_kappa(op_v[a], op_v[b]) for a, b in nli_pairs)
        reversed_here = op_llm > op_nli
        n_ops_reversed += 1 if reversed_here else 0
        marker = " (reversed)" if reversed_here else " (not reversed)"
        print(f"    {op:22s} within-LLM={op_llm:+.3f}  within-NLI={op_nli:+.3f}{marker}")
    print(f"  Reversal holds on {n_ops_reversed}/{len(adversarial_ops)} adversarial operators")
    d3_pass = n_ops_reversed == len(adversarial_ops)
    print(f"  D3 {'PASS' if d3_pass else 'FAIL'} (reversal must hold for ALL adversarial ops)")

    # ================================================================
    # E1: ARCHITECTURE × OPERATOR FNR MATRIX
    # ================================================================
    print()
    print("=" * 78)
    print("E1: Architecture × operator FNR (architectural blind spots)")
    print("=" * 78)
    print()
    print(f"{'operator':22s} | {'LLM avg':>8s} | {'NLI avg':>8s} | {'RAGAS':>8s}")
    print("-" * 60)
    for op in adversarial_ops:
        op_cells = [(s, o) for (s, o) in full_cells if o == op]
        llm_fnrs = []
        for j in llm_judges:
            n = sum(1 for (s, o) in op_cells if verdict_map[(s, op, j)] == "faithful")
            llm_fnrs.append(n / max(1, len(op_cells)))
        nli_fnrs = []
        for j in nli_judges:
            n = sum(1 for (s, o) in op_cells if verdict_map[(s, op, j)] == "faithful")
            nli_fnrs.append(n / max(1, len(op_cells)))
        ragas_fnr = sum(1 for (s, o) in op_cells if verdict_map[(s, op, "ragas_style_sonnet")] == "faithful") / max(1, len(op_cells))
        print(f"{op:22s} |  {100*mean(llm_fnrs):5.1f}%  |  {100*mean(nli_fnrs):5.1f}%  |  {100*ragas_fnr:5.1f}%")

    # ================================================================
    # E2: ENSEMBLE SHRINKAGE — smallest set that approximates full ensemble
    # ================================================================
    print()
    print("=" * 78)
    print("E2: Ensemble shrinkage — can we approximate the 7-judge ensemble?")
    print("=" * 78)
    # Define "full ensemble verdict" as majority of 7 judges.
    # Then ask: what's the minimum subset that matches the full ensemble's verdict?

    def majority(vs, default="faithful"):
        n_faith = vs.count("faithful")
        n_unfaith = vs.count("unfaithful")
        if n_faith > n_unfaith:
            return "faithful"
        elif n_unfaith > n_faith:
            return "unfaithful"
        return default

    full_ens = {}
    for sid, op in full_cells:
        if op == "paraphrase_null":
            continue
        vs = [verdict_map[(sid, op, j)] for j in judges]
        full_ens[(sid, op)] = majority(vs)

    # Try every subset of size 2, 3, 4 of the 7 judges and report best agreement with full ensemble
    cells_adv = [(s, o) for (s, o) in full_cells if o != "paraphrase_null"]
    print(f"\n  Testing all subsets of size 2-4 (out of {len(judges)} judges) for agreement with majority-of-7 ensemble verdict")
    print(f"  Comparison set: {len(cells_adv)} adversarial cells")
    print()
    for size in (2, 3, 4):
        best = None
        for subset in combinations(judges, size):
            agree = 0
            for (sid, op), full_v in full_ens.items():
                sub_vs = [verdict_map[(sid, op, j)] for j in subset]
                sub_v = majority(sub_vs)
                if sub_v == full_v:
                    agree += 1
            pct = agree / max(1, len(full_ens))
            if best is None or pct > best[1]:
                best = (subset, pct)
        print(f"  Best {size}-judge subset: {best[0]} → matches majority-of-7 on {100*best[1]:.1f}% of cells")

    # ================================================================
    # E3: ALIGNSCORE STANDALONE
    # ================================================================
    print()
    print("=" * 78)
    print("E3: AlignScore standalone behavior")
    print("=" * 78)
    if "alignscore_large" in judges:
        n_total = 0
        n_faithful = 0
        for sid, op in full_cells:
            n_total += 1
            if verdict_map[(sid, op, "alignscore_large")] == "faithful":
                n_faithful += 1
        # also check clean verdicts
        n_clean_faithful = sum(1 for v in verdicts
                               if v["operator"] in CLEAN_OPS and v["judge"] == "alignscore_large"
                               and v["verdict"] == "faithful")
        n_clean_total = sum(1 for v in verdicts
                            if v["operator"] in CLEAN_OPS and v["judge"] == "alignscore_large")
        print(f"  AlignScore says 'faithful' on {n_faithful}/{n_total} = {100*n_faithful/max(1,n_total):.1f}% of perturbed cells")
        print(f"  AlignScore says 'faithful' on {n_clean_faithful}/{n_clean_total} = {100*n_clean_faithful/max(1,n_clean_total):.1f}% of CLEAN cells")
        print(f"  → AlignScore appears to be near-uniform: ratio perturbed/clean = {(n_faithful/max(1,n_total)) / max(0.01, n_clean_faithful/max(1,n_clean_total)):.3f}")
        print(f"  → If ratio ~1.0, AlignScore is not discriminating between clean and perturbed answers.")


if __name__ == "__main__":
    main()
