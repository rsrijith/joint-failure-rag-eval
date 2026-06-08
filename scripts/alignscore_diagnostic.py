"""Diagnose whether AlignScore-large is salvageable on this task.

The default 0.5 threshold makes AlignScore call ~0% of clean gold answers
faithful. Before dropping it we check WHY:
  (1) threshold miscalibration  -> raw scores separate clean vs perturbed but
      the 0.5 cut is wrong;  recalibrate.
  (2) long-premise truncation   -> scores collapse when premise > 512 tokens;
      harness bug, fixable.
  (3) genuine non-discrimination -> clean and perturbed score distributions
      overlap; report as a finding.

Re-runs AlignScore on a sample, capturing the raw continuous score, and
reports: clean vs perturbed score distributions, ROC-AUC, Youden-optimal
threshold, and score-vs-premise-length correlation.

Local model, no API cost.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from jfre.judges import alignscore_judge
from jfre.judges._llm_judge_prompt import format_passages
from jfre.seeds.expertqa import load as load_expertqa
from jfre.seeds.hotpotqa import load as load_hotpotqa

RNG = random.Random(7)
N_CLEAN = 60
N_PER_OP = 25
ADVERSARIAL_OPS = ["entity_swap", "numeric_drift", "hedge_insertion",
                   "distractor_parroting", "citation_relocation"]


def build_seed_map():
    smap = {}
    for s in load_hotpotqa(n=1000):
        smap[s.seed_id] = s
    for s in load_expertqa(n=1500):
        smap[s.seed_id] = s
    return smap


def load_perts():
    out = {}
    for base in ("results/preview_pilot/perturbations.jsonl",
                 "results/citation_relocation_pilot/perturbations.jsonl"):
        p = Path(base)
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r.get("rule_passed"):
                out[(r["seed_id"], r["operator"])] = r
    return out


def accepted_seed_ids():
    ids = set()
    for line in Path("results/preview_pilot/seeds.jsonl").open():
        s = json.loads(line)
        if s.get("accepted"):
            ids.add(s["seed_id"])
    for line in Path("results/citation_relocation_pilot/seeds.jsonl").open():
        s = json.loads(line)
        if s.get("accepted"):
            ids.add(s["seed_id"])
    return ids


def raw_score(seed, answer):
    """Call AlignScore, return (raw_score, premise_token_estimate)."""
    v = alignscore_judge.score(seed, answer, operator="clean")
    premise = format_passages(seed.passages)
    approx_tokens = len(premise.split())
    return v.raw_score, approx_tokens, v.judge_metadata.get("error")


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    def q(p):
        return xs[min(n - 1, int(p * n))]
    return {"min": xs[0], "p25": q(0.25), "median": q(0.5),
            "p75": q(0.75), "max": xs[-1], "mean": sum(xs) / n}


def roc_auc(pos_scores, neg_scores):
    """AUC = P(random clean score > random perturbed score). pos=clean (faithful)."""
    if not pos_scores or not neg_scores:
        return None
    wins = ties = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores))


def youden_threshold(pos_scores, neg_scores):
    """Threshold maximizing (TPR - FPR), pos=clean should be ABOVE threshold."""
    cand = sorted(set(pos_scores + neg_scores))
    best_t, best_j = 0.5, -1
    for t in cand:
        tpr = sum(1 for s in pos_scores if s >= t) / len(pos_scores)
        fpr = sum(1 for s in neg_scores if s >= t) / len(neg_scores)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t = j, t
    return best_t, best_j


def main():
    smap = build_seed_map()
    perts = load_perts()
    accepted = accepted_seed_ids()

    print("Running AlignScore with raw-score capture (local, no API)...\n")

    # Clean sample
    clean_ids = [sid for sid in accepted if sid in smap]
    RNG.shuffle(clean_ids)
    clean_ids = clean_ids[:N_CLEAN]
    clean_scores, clean_lens = [], []
    for sid in clean_ids:
        sc, ln, err = raw_score(smap[sid], smap[sid].gold_answer)
        if err or sc is None:
            continue
        clean_scores.append(sc)
        clean_lens.append(ln)

    # Perturbed sample per operator
    pert_scores = defaultdict(list)
    pert_lens = []
    for op in ADVERSARIAL_OPS:
        cells = [(sid, o) for (sid, o) in perts if o == op and sid in smap]
        RNG.shuffle(cells)
        for sid, o in cells[:N_PER_OP]:
            sc, ln, err = raw_score(smap[sid], perts[(sid, o)]["perturbed_answer"])
            if err or sc is None:
                continue
            pert_scores[op].append(sc)
            pert_lens.append((ln, sc))

    print("=" * 70)
    print("  AlignScore raw-score distributions (threshold currently 0.5)")
    print("=" * 70)
    print(f"\n  CLEAN gold answers (should score HIGH):")
    cq = quantiles(clean_scores)
    if cq:
        print(f"    n={len(clean_scores)}  mean={cq['mean']:.3f}  "
              f"min={cq['min']:.3f} p25={cq['p25']:.3f} median={cq['median']:.3f} "
              f"p75={cq['p75']:.3f} max={cq['max']:.3f}")

    print(f"\n  PERTURBED answers (should score LOW):")
    all_pert = []
    for op in ADVERSARIAL_OPS:
        pq = quantiles(pert_scores[op])
        all_pert.extend(pert_scores[op])
        if pq:
            print(f"    {op:<22} n={len(pert_scores[op]):>3}  mean={pq['mean']:.3f}  "
                  f"median={pq['median']:.3f}  [{pq['min']:.3f}, {pq['max']:.3f}]")

    print("\n" + "=" * 70)
    print("  Discrimination analysis")
    print("=" * 70)
    auc = roc_auc(clean_scores, all_pert)
    print(f"\n  ROC-AUC (clean vs all-perturbed) = {auc:.3f}" if auc else "  AUC n/a")
    print("    0.5 = no discrimination, 1.0 = perfect, <0.5 = inverted")
    t, j = youden_threshold(clean_scores, all_pert)
    print(f"  Youden-optimal threshold = {t:.3f} (J = {j:.3f})")
    if clean_scores and all_pert:
        tpr = sum(1 for s in clean_scores if s >= t) / len(clean_scores)
        fpr = sum(1 for s in all_pert if s >= t) / len(all_pert)
        print(f"    at that threshold: clean-faithful={tpr*100:.0f}%, perturbed-faithful(FNR)={fpr*100:.0f}%")
        tpr5 = sum(1 for s in clean_scores if s >= 0.5) / len(clean_scores)
        fpr5 = sum(1 for s in all_pert if s >= 0.5) / len(all_pert)
        print(f"    at default 0.5:    clean-faithful={tpr5*100:.0f}%, perturbed-faithful(FNR)={fpr5*100:.0f}%")

    print("\n" + "=" * 70)
    print("  Long-premise effect (RoBERTa 512-token limit)")
    print("=" * 70)
    if clean_lens and clean_scores:
        short = [s for s, l in zip(clean_scores, clean_lens) if l <= 350]
        longp = [s for s, l in zip(clean_scores, clean_lens) if l > 350]
        print(f"\n  CLEAN scores by premise length (~words):")
        sq, lq = quantiles(short), quantiles(longp)
        if sq:
            print(f"    premise <=350w: n={len(short)} mean={sq['mean']:.3f} median={sq['median']:.3f}")
        if lq:
            print(f"    premise  >350w: n={len(longp)} mean={lq['mean']:.3f} median={lq['median']:.3f}")
        print("    If long-premise mean is much lower, truncation is hurting AlignScore.")

    print("\n  VERDICT GUIDE:")
    print("    - AUC >= 0.75 and Youden threshold gives reasonable clean/FNR -> RECALIBRATE, keep it.")
    print("    - long-premise mean << short-premise mean -> FIX CHUNKING, re-run.")
    print("    - AUC ~ 0.5 regardless -> genuinely non-discriminative -> report as finding.")


if __name__ == "__main__":
    main()
