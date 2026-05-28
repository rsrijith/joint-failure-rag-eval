"""Hypothesis validation: should we ship this paper?

Runs the paper's load-bearing hypotheses against the collected pilot
data and outputs a go / framing-pivot / no-go recommendation. Designed
to be run BEFORE drafting, so we don't burn writing-and-attorney time
on a story the data doesn't support.

Hypotheses tested (each gets PASS / WEAK / FAIL):

  H1. Joint failure exists at non-trivial rate
      - STRONG: ≥10% on at least one adversarial operator
      - WEAK: 2-10% (statistically detectable but not headline-grade)
      - FAIL: <2% across all adversarial operators

  H2. Joint failure is correlated, not independent
      - PASS: attack universality > +2pp on at least one operator
        (i.e. observed joint failure exceeds product-of-marginals baseline)
      - FAIL: universality ≤ 0 (judges fail independently)

  H3. Architectural clustering in pairwise Cohen's kappa
      - STRONG: within-cluster average κ ≥ 0.5, across-cluster average ≤ 0.3
      - WEAK: within-cluster > across-cluster but gap < 0.2
      - FAIL: no within-vs-across separation

  H4. Per-operator variation: different operators have different signatures
      - PASS: range(joint-failure across adversarial ops) ≥ 5pp OR
              range(judge-cluster κ across ops) ≥ 0.2
      - FAIL: all operators behave the same

  H5. AlignScore sanity check (mid-pilot showed 91-100% FNR — possibly miscalibrated)
      - PASS: AlignScore FNR < 80% on at least one adversarial operator
      - WARN: AlignScore FNR ≥ 80% on all operators (judge possibly broken
              for our setting; may need to be reported separately or dropped)

Final verdict:
  STRONG GO:    H1 PASS or WEAK + H2 PASS + H3 STRONG + H4 PASS → ship as-framed
  PIVOT GO:     H2 PASS + H3 PASS (any strength) + H4 PASS → reframe story to
                emphasize correlation/clustering over absolute joint failure
  WEAK GO:      H2 PASS + (H3 WEAK or H4 PASS) → small workshop, narrow framing
  NO GO:        H2 FAIL OR (H3 FAIL AND H4 FAIL) → no publishable story
                in current frame; consider re-scoping

Run from repo root:
    python scripts/hypothesis_check.py
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

OUT_DIRS = [Path("results/preview_pilot"), Path("results/citation_relocation_pilot")]
CLEAN_OPS = {"clean", "clean_cited"}
EXCLUDED_JUDGES = {"glm_4_7_cerebras", "qwen3_235b_cerebras", "claude_opus_4_7"}

LLM_CLUSTER = {"claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"}
NLI_CLUSTER = {"hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large"}
DECOMP_CLUSTER = {"ragas_style_sonnet"}


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


def _load_all():
    verdicts, perturbations = [], []
    for d in OUT_DIRS:
        v_path = d / "verdicts.jsonl"
        p_path = d / "perturbations.jsonl"
        if v_path.exists():
            verdicts.extend(json.loads(line) for line in v_path.open())
        if p_path.exists():
            perturbations.extend(json.loads(line) for line in p_path.open())
    return verdicts, perturbations


def main():
    verdicts, perturbations = _load_all()
    if not verdicts:
        raise SystemExit("No verdicts loaded.")

    judges = sorted({
        v["judge"] for v in verdicts
        if v["operator"] not in CLEAN_OPS
        and v["judge"] not in EXCLUDED_JUDGES
    })
    operators = sorted({
        v["operator"] for v in verdicts if v["operator"] not in CLEAN_OPS
    })
    adversarial_ops = [op for op in operators if op != "paraphrase_null"]

    valid_perturbed = {(p["seed_id"], p["operator"]) for p in perturbations if p["rule_passed"]}
    verdict_map = {(v["seed_id"], v["operator"], v["judge"]): v["verdict"] for v in verdicts}

    print("=" * 78)
    print("HYPOTHESIS VALIDATION")
    print("=" * 78)
    print(f"Judges ({len(judges)}): {judges}")
    print(f"Operators ({len(operators)}): {operators}")
    print(f"Valid perturbed cells: {len(valid_perturbed)}")
    print()

    # ---- H1 + H2 ----
    print("-" * 78)
    print("H1: Joint failure exists at non-trivial rate")
    print("H2: Joint failure is correlated (attack universality > 0)")
    print("-" * 78)
    h1_strong = h1_weak = False
    h2_pass = False
    max_jfr = 0.0
    max_universality = 0.0
    op_results = []
    for op in adversarial_ops:
        n_joint_fail = 0
        n_eval = 0
        for sid, oo in valid_perturbed:
            if oo != op:
                continue
            row = [verdict_map.get((sid, op, j)) for j in judges]
            if any(v is None for v in row):
                continue
            n_eval += 1
            if all(v == "faithful" for v in row):
                n_joint_fail += 1
        jfr = (n_joint_fail / max(1, n_eval))
        marginals = []
        for j in judges:
            n_fnr = sum(1 for sid, oo in valid_perturbed
                        if oo == op and verdict_map.get((sid, op, j)) == "faithful")
            n_tot = sum(1 for sid, oo in valid_perturbed if oo == op)
            marginals.append(n_fnr / max(1, n_tot))
        baseline = 1.0
        for m in marginals:
            baseline *= m
        universality = jfr - baseline
        op_results.append((op, jfr, baseline, universality, n_eval))
        max_jfr = max(max_jfr, jfr)
        max_universality = max(max_universality, universality)
        if jfr >= 0.10:
            h1_strong = True
        elif jfr >= 0.02:
            h1_weak = True
        if universality > 0.02:
            h2_pass = True
        print(f"  {op:22s} JFR={100*jfr:5.1f}%  baseline={100*baseline:5.2f}%  universality={100*universality:+.2f}pp  (n={n_eval})")
    print()
    h1_verdict = "STRONG" if h1_strong else ("WEAK" if h1_weak else "FAIL")
    h2_verdict = "PASS" if h2_pass else "FAIL"
    print(f"  H1 verdict: {h1_verdict}  (max JFR = {100*max_jfr:.1f}%)")
    print(f"  H2 verdict: {h2_verdict}  (max universality = {100*max_universality:+.2f}pp)")
    print()

    # ---- H3: architectural clustering ----
    print("-" * 78)
    print("H3: Architectural clustering in pairwise Cohen's kappa")
    print("-" * 78)
    per_judge_v: dict[str, list[str]] = {j: [] for j in judges}
    for sid, op in valid_perturbed:
        row = {j: verdict_map.get((sid, op, j)) for j in judges}
        if any(v is None for v in row.values()):
            continue
        for j, v in row.items():
            per_judge_v[j].append(v)
    kappa_pair: dict[tuple, float] = {}
    for i, j1 in enumerate(judges):
        for j2 in judges[i + 1:]:
            kappa_pair[(j1, j2)] = _cohens_kappa(per_judge_v[j1], per_judge_v[j2])
    within_llm = [k for (a, b), k in kappa_pair.items()
                  if a in LLM_CLUSTER and b in LLM_CLUSTER]
    within_nli = [k for (a, b), k in kappa_pair.items()
                  if a in NLI_CLUSTER and b in NLI_CLUSTER]
    across = [k for (a, b), k in kappa_pair.items()
              if (a in LLM_CLUSTER) != (b in LLM_CLUSTER)
              and (a in NLI_CLUSTER) != (b in NLI_CLUSTER)
              or ({a, b} & LLM_CLUSTER and {a, b} & NLI_CLUSTER)]
    # Cleaner across-cluster: any pair where one is LLM and the other is NLI
    across = [k for (a, b), k in kappa_pair.items()
              if (a in LLM_CLUSTER and b in NLI_CLUSTER)
              or (a in NLI_CLUSTER and b in LLM_CLUSTER)]
    llm_avg = mean(within_llm) if within_llm else 0.0
    nli_avg = mean(within_nli) if within_nli else 0.0
    across_avg = mean(across) if across else 0.0
    print(f"  Within LLM-cluster κ avg:   {llm_avg:+.3f}  (n_pairs={len(within_llm)})")
    print(f"  Within NLI-cluster κ avg:   {nli_avg:+.3f}  (n_pairs={len(within_nli)})")
    print(f"  Across LLM-vs-NLI κ avg:    {across_avg:+.3f}  (n_pairs={len(across)})")
    within_avg = mean(within_llm + within_nli) if (within_llm + within_nli) else 0.0
    gap = within_avg - across_avg
    print(f"  Within - across gap:        {gap:+.3f}")
    if within_avg >= 0.5 and across_avg <= 0.3:
        h3 = "STRONG"
    elif gap >= 0.2:
        h3 = "PASS"
    elif gap > 0.05:
        h3 = "WEAK"
    else:
        h3 = "FAIL"
    print(f"  H3 verdict: {h3}")
    print()

    # ---- H4: per-operator variation ----
    print("-" * 78)
    print("H4: Per-operator variation")
    print("-" * 78)
    jfrs = [jfr for _, jfr, _, _, _ in op_results]
    jfr_range = (max(jfrs) - min(jfrs)) if jfrs else 0.0
    print(f"  Joint-failure range across adversarial ops: {100*jfr_range:.1f}pp")
    h4 = "PASS" if jfr_range >= 0.05 else "FAIL"
    print(f"  H4 verdict: {h4}")
    print()

    # ---- H5: AlignScore sanity ----
    print("-" * 78)
    print("H5: AlignScore sanity (mid-pilot showed 91-100% FNR)")
    print("-" * 78)
    if "alignscore_large" not in judges:
        print("  AlignScore not in current judges set. SKIP.")
        h5 = "SKIP"
    else:
        as_fnrs = []
        for op in adversarial_ops:
            n_fnr = sum(1 for sid, oo in valid_perturbed
                        if oo == op and verdict_map.get((sid, op, "alignscore_large")) == "faithful")
            n_tot = sum(1 for sid, oo in valid_perturbed if oo == op)
            if n_tot > 0:
                as_fnrs.append((op, n_fnr / n_tot))
        for op, fnr in as_fnrs:
            print(f"  {op:22s} AlignScore FNR = {100*fnr:.1f}%")
        min_as_fnr = min(fnr for _, fnr in as_fnrs) if as_fnrs else 1.0
        h5 = "PASS" if min_as_fnr < 0.80 else "WARN"
        print(f"  H5 verdict: {h5}  (min FNR across adversarial ops = {100*min_as_fnr:.1f}%)")
    print()

    # ---- Final recommendation ----
    print("=" * 78)
    print("FINAL VERDICT")
    print("=" * 78)
    print(f"  H1: {h1_verdict}   H2: {h2_verdict}   H3: {h3}   H4: {h4}   H5: {h5}")
    print()

    if (h1_verdict in {"STRONG", "WEAK"}) and h2_verdict == "PASS" and h3 == "STRONG" and h4 == "PASS":
        verdict = "STRONG GO — ship as originally framed (joint failure across deployed judges)"
    elif h2_verdict == "PASS" and h3 in {"STRONG", "PASS"} and h4 == "PASS":
        verdict = "PIVOT GO — reframe to emphasize architectural clustering / correlated failure, not absolute joint failure rate"
    elif h2_verdict == "PASS" and (h3 != "FAIL" or h4 == "PASS"):
        verdict = "WEAK GO — narrow workshop framing; consider whether the contribution clears the venue bar"
    elif h2_verdict == "FAIL" and h3 == "FAIL" and h4 == "FAIL":
        verdict = "NO GO — data does not support the paper's claims in any framing"
    else:
        verdict = "MIXED — discuss before drafting; some hypotheses hold and others don't"

    print(f"  {verdict}")

    if h5 == "WARN":
        print()
        print("  NOTE: AlignScore appears miscalibrated or insensitive for this task.")
        print("        Consider reporting separately, dropping from the headline 7-judge")
        print("        ensemble (down to 6), or treating its consistent 'faithful' votes")
        print("        as a baseline rather than a judge signal.")


if __name__ == "__main__":
    main()
