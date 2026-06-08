"""Confidence intervals + significance tests for the headline citation finding.

- Wilson 95% CI on every per-judge FNR (content + attribution prompt).
- Bootstrap 95% CI on the attribution Δ (content FNR - attribution FNR), per judge.
- McNemar exact paired test: does the attribution prompt change verdicts on the same
  scrambled cells? (b = content-faithful->attribution-unfaithful, c = the reverse).
- Same for the Llama-annotated cross-model run.
- Wilson CIs for the full 7-judge x 6-operator FNR table (caption numbers).
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

LLM = ["claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"]
ALL7 = LLM + ["hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large", "ragas_style_sonnet"]
RNG = random.Random(20260603)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - half) / d, (c + half) / d)


def load_map(path, op_filter=None, key=("seed_id", "judge"), val="verdict"):
    out = {}
    for line in Path(path).open():
        r = json.loads(line)
        if op_filter and r.get("operator") != op_filter:
            continue
        if r.get("metadata", {}).get("error") or r.get(val) not in ("faithful", "unfaithful"):
            continue
        out[(r["seed_id"], r["judge"])] = r[val]
    return out


def fnr_block(label, content, attr_scrambled, judges):
    """content/attr maps: (seed,judge)->verdict on SCRAMBLED citations. FNR = faithful%."""
    print(f"\n{'='*78}\n  {label}\n{'='*78}")
    print(f"  {'judge':<24} {'content FNR (95% CI)':>26} {'attr FNR (95% CI)':>22} {'Δ (boot 95% CI)':>20} {'McNemar p':>10}")
    for j in judges:
        seeds = sorted({s for (s, jj) in content if jj == j} & {s for (s, jj) in attr_scrambled if jj == j})
        n = len(seeds)
        if n == 0:
            continue
        c_faith = [content[(s, j)] == "faithful" for s in seeds]
        a_faith = [attr_scrambled[(s, j)] == "faithful" for s in seeds]
        cf, af = sum(c_faith), sum(a_faith)
        c_lo, c_hi = wilson(cf, n)
        a_lo, a_hi = wilson(af, n)
        # bootstrap Δ (content - attribution FNR)
        deltas = []
        for _ in range(5000):
            idx = [RNG.randrange(n) for _ in range(n)]
            deltas.append(sum(c_faith[i] for i in idx) / n - sum(a_faith[i] for i in idx) / n)
        deltas.sort()
        d_lo, d_hi = deltas[int(0.025 * 5000)], deltas[int(0.975 * 5000)]
        delta = cf / n - af / n
        # McNemar: discordant pairs
        b = sum(1 for i in range(n) if c_faith[i] and not a_faith[i])  # content-miss fixed by attr
        cc = sum(1 for i in range(n) if not c_faith[i] and a_faith[i])  # attr-miss
        p = binomtest(min(b, cc), b + cc, 0.5).pvalue if (b + cc) > 0 else 1.0
        print(f"  {j:<24} {cf*100/n:>5.0f}% [{c_lo*100:>3.0f},{c_hi*100:>3.0f}]  (n={n}) "
              f"{af*100/n:>5.0f}% [{a_lo*100:>3.0f},{a_hi*100:>3.0f}] "
              f"{delta*100:>+5.0f}pp [{d_lo*100:>+3.0f},{d_hi*100:>+3.0f}] {p:>10.2e}")


def main():
    base = "results/citation_relocation_pilot"
    # content prompt = deployed citation_relocation verdicts; attribution = the experiment
    content = load_map(f"{base}/verdicts.jsonl", op_filter="citation_relocation")
    attr = load_map(f"{base}/verdicts_attribution_prompt.jsonl")
    fnr_block("Claude-annotated citations (content prompt vs attribution prompt, scrambled)",
              content, attr, LLM)

    # cross-model (Llama-annotated): content + attribution both from the crossmodel file
    cm = defaultdict(dict)
    for line in Path(f"{base}/verdicts_crossmodel_attribution.jsonl").open():
        r = json.loads(line)
        if r["verdict"] in ("faithful", "unfaithful"):
            cm[(r["condition"], r["prompt_type"])][(r["seed_id"], r["judge"])] = r["verdict"]
    cm_content = cm[("scrambled", "content")]
    cm_attr = cm[("scrambled", "attribution")]
    fnr_block("Llama-annotated citations (cross-model control, scrambled)",
              cm_content, cm_attr, LLM)

    # Wilson CIs for the full §5.1 FNR table (per judge per operator)
    print(f"\n{'='*78}\n  §5.1 FNR table with Wilson 95% CI (per judge x operator)\n{'='*78}")
    cells = defaultdict(lambda: [0, 0])  # (op, judge) -> [faithful, total]
    for fpath, ops in [("results/preview_pilot/verdicts.jsonl", None),
                       (f"{base}/verdicts.jsonl", None)]:
        for line in Path(fpath).open():
            r = json.loads(line)
            op, j = r["operator"], r["judge"]
            if j not in ALL7 or op in ("clean", "clean_cited"):
                continue
            if r.get("metadata", {}).get("error") or r["verdict"] not in ("faithful", "unfaithful"):
                continue
            cells[(op, j)][1] += 1
            if r["verdict"] == "faithful":
                cells[(op, j)][0] += 1
    OPS = ["entity_swap", "numeric_drift", "hedge_insertion", "distractor_parroting",
           "citation_relocation", "paraphrase_null"]
    for op in OPS:
        ns = [cells[(op, j)][1] for j in ALL7 if cells[(op, j)][1]]
        n = ns[0] if ns else 0
        parts = []
        for j in ALL7:
            f, t = cells[(op, j)]
            if t:
                lo, hi = wilson(f, t)
                parts.append(f"{j.split('_')[0][:5]}:{f*100//t}[{lo*100:.0f},{hi*100:.0f}]")
        print(f"  {op:<22} (n={n}): " + "  ".join(parts))


if __name__ == "__main__":
    main()
