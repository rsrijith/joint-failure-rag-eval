"""Reviewer-driven re-analyses that need no model calls (pure label-level).

Three analyses:
  1. Per-judge CLEAN faithful rate. A well-behaved judge should call a high
     fraction of clean (gold) answers faithful. A judge near 0% is degenerate
     (calls everything unfaithful) and its low cross-judge kappa / its "0% FNR"
     on adversarial operators is an artifact of being a near-constant predictor.

  2. Pre-filter re-derivation. The >=4/7 accept filter ran while AlignScore was
     broken (defaulting to 'faithful'). Recompute the accept set with the
     clean-pass AlignScore verdicts, plus an AlignScore-excluded variant, and
     compare to the originally-accepted seed set (Jaccard).

  3. Per-judge faithful-rate by condition (clean vs each operator). Surfaces
     which judges discriminate (high on clean, low on adversarial, high on the
     paraphrase_null negative control) vs which are near-constant.

Reads results/preview_pilot/verdicts.jsonl and results/citation_relocation_pilot/verdicts.jsonl.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HEADLINE_JUDGES = [
    "claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet",
    "hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large",
    "ragas_style_sonnet",
]

LLM = {"claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"}
NLI = {"hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large"}
DECOMP = {"ragas_style_sonnet"}

ADVERSARIAL_OPS = ["entity_swap", "numeric_drift", "hedge_insertion",
                   "distractor_parroting", "citation_relocation"]
CONTROL_OPS = ["paraphrase_null"]
CLEAN_OPS = ["clean", "clean_cited"]


def load_verdicts():
    """Return list of verdict dicts (non-error, headline judges only)."""
    rows = []
    for base in ("results/preview_pilot/verdicts.jsonl",
                 "results/citation_relocation_pilot/verdicts.jsonl"):
        p = Path(base)
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r["judge"] not in HEADLINE_JUDGES:
                continue
            if r.get("metadata", {}).get("error"):
                continue
            rows.append(r)
    return rows


def faithful_rate(rows, judge, ops):
    sub = [r for r in rows if r["judge"] == judge and r["operator"] in ops]
    if not sub:
        return None, 0
    n_faith = sum(1 for r in sub if r["verdict"] == "faithful")
    return n_faith / len(sub), len(sub)


def analysis_1_and_3(rows):
    print("=" * 78)
    print("  ANALYSIS 1 + 3: Per-judge faithful rate by condition")
    print("=" * 78)
    print()
    print("  KEY: clean -> want HIGH (gold IS faithful)")
    print("       adversarial ops -> want LOW (perturbed is unfaithful; high = FNR)")
    print("       paraphrase_null -> want HIGH (negative control, still faithful)")
    print()
    header = f"  {'judge':<26} {'CLEAN':>10} | " + " ".join(f"{op[:9]:>10}" for op in ADVERSARIAL_OPS) + f" | {'paraph_null':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for judge in HEADLINE_JUDGES:
        clean_rate, clean_n = faithful_rate(rows, judge, CLEAN_OPS)
        clean_str = f"{clean_rate*100:5.0f}%({clean_n:>3})" if clean_rate is not None else "   n/a"
        op_strs = []
        for op in ADVERSARIAL_OPS:
            r, n = faithful_rate(rows, judge, [op])
            op_strs.append(f"{r*100:5.0f}%({n:>3})" if r is not None else "   n/a   ")
        ctrl_rate, ctrl_n = faithful_rate(rows, judge, CONTROL_OPS)
        ctrl_str = f"{ctrl_rate*100:5.0f}%({ctrl_n:>3})" if ctrl_rate is not None else "   n/a"
        print(f"  {judge:<26} {clean_str:>10} | " + " ".join(f"{s:>10}" for s in op_strs) + f" | {ctrl_str:>12}")

    print()
    print("  DEGENERACY FLAG: a judge with CLEAN faithful rate < 20% is near-constant-")
    print("  unfaithful. Its '0% FNR' on adversarial ops is then not discrimination but")
    print("  the constant happening to match the unfaithful gold label.")
    print()
    for judge in HEADLINE_JUDGES:
        clean_rate, clean_n = faithful_rate(rows, judge, CLEAN_OPS)
        if clean_rate is not None and clean_rate < 0.20:
            # what's its overall faithful rate across everything?
            all_rate, all_n = faithful_rate(rows, judge, CLEAN_OPS + ADVERSARIAL_OPS + CONTROL_OPS)
            print(f"    >>> {judge}: clean faithful={clean_rate*100:.0f}% (n={clean_n}); "
                  f"overall faithful={all_rate*100:.0f}% across all conditions -> DEGENERATE")


def analysis_2_prefilter(rows):
    print()
    print("=" * 78)
    print("  ANALYSIS 2: Pre-filter re-derivation with clean AlignScore")
    print("=" * 78)
    print()
    # Build clean verdicts per seed per judge (main pilot only; clean op)
    clean_verdicts = defaultdict(dict)  # seed_id -> {judge: verdict}
    for r in rows:
        if r["operator"] == "clean":
            clean_verdicts[r["seed_id"]][r["judge"]] = r["verdict"]

    # Originally-accepted seeds (from seeds.jsonl)
    orig_accepted = set()
    orig_all = set()
    for line in Path("results/preview_pilot/seeds.jsonl").open():
        s = json.loads(line)
        orig_all.add(s["seed_id"])
        if s.get("accepted"):
            orig_accepted.add(s["seed_id"])

    # Recompute accept under three rules. Only consider seeds that have clean
    # verdicts from all 7 judges (so the comparison is apples-to-apples).
    full_clean_seeds = {sid for sid, jv in clean_verdicts.items()
                        if all(j in jv for j in HEADLINE_JUDGES)}

    def n_faithful(jv, judges):
        return sum(1 for j in judges if jv.get(j) == "faithful")

    rule_all7_ge4 = set()       # >=4 of 7 (the original rule, now with clean AlignScore)
    rule_no_align_ge4 = set()   # >=4 of the 6 non-AlignScore judges
    rule_no_align_ge3 = set()   # >=3 of the 6 non-AlignScore judges (majority of 6)
    six = [j for j in HEADLINE_JUDGES if j != "alignscore_large"]
    for sid in full_clean_seeds:
        jv = clean_verdicts[sid]
        if n_faithful(jv, HEADLINE_JUDGES) >= 4:
            rule_all7_ge4.add(sid)
        if n_faithful(jv, six) >= 4:
            rule_no_align_ge4.add(sid)
        if n_faithful(jv, six) >= 3:
            rule_no_align_ge3.add(sid)

    def jaccard(a, b):
        return len(a & b) / len(a | b) if (a | b) else 1.0

    print(f"  Seeds with all-7 clean verdicts (comparison base): {len(full_clean_seeds)}")
    print(f"  Originally accepted (any rule, broken AlignScore):  {len(orig_accepted)}")
    print(f"  Originally accepted AND in full-clean base:         {len(orig_accepted & full_clean_seeds)}")
    print()
    print(f"  Re-derived accept sets (on the {len(full_clean_seeds)}-seed full-clean base):")
    print(f"    >=4 of 7 (clean AlignScore):       {len(rule_all7_ge4)}")
    print(f"    >=4 of 6 (AlignScore excluded):    {len(rule_no_align_ge4)}")
    print(f"    >=3 of 6 (majority, AlignScore ex):{len(rule_no_align_ge3)}")
    print()
    base_orig = orig_accepted & full_clean_seeds
    print(f"  Jaccard(original-accept, re-derived) on full-clean base:")
    print(f"    vs >=4 of 7 (clean AlignScore):    {jaccard(base_orig, rule_all7_ge4):.3f}  "
          f"(dropped {len(base_orig - rule_all7_ge4)}, added {len(rule_all7_ge4 - base_orig)})")
    print(f"    vs >=4 of 6 (AlignScore excluded): {jaccard(base_orig, rule_no_align_ge4):.3f}  "
          f"(dropped {len(base_orig - rule_no_align_ge4)}, added {len(rule_no_align_ge4 - base_orig)})")
    print(f"    vs >=3 of 6 (majority, no Align):  {jaccard(base_orig, rule_no_align_ge3):.3f}  "
          f"(dropped {len(base_orig - rule_no_align_ge3)}, added {len(rule_no_align_ge3 - base_orig)})")
    print()
    print("  INTERPRETATION: if Jaccard is high (>0.90) the broken-AlignScore filter")
    print("  did not materially change which seeds entered Phase 2 -> one-line robustness")
    print("  note suffices. If low, headline numbers must be recomputed on the intersection.")


def main():
    rows = load_verdicts()
    print(f"Loaded {len(rows)} non-error headline-judge verdicts.\n")
    analysis_1_and_3(rows)
    analysis_2_prefilter(rows)


if __name__ == "__main__":
    main()
