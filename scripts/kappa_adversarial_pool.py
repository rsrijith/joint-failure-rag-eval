"""F3 fix: full 7x7 pairwise Cohen's kappa on the SINGLE adversarial pool (1213
cells), so the paper's Table 2 and its within/between-cluster contrast read off
one pool instead of two (the old Table 2 was the 1483 all-operator pool).

Adversarial = the 5 real operators (excludes the paraphrase_null negative control
and any clean cells). Keeps only cells where all 7 headline judges have a valid
faithful/unfaithful verdict.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

JUDGES = ["claude_sonnet_4_6", "faithjudge_style_sonnet", "mistral_large_2",
          "hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large", "ragas_style_sonnet"]
LLM = {"claude_sonnet_4_6", "faithjudge_style_sonnet", "mistral_large_2"}
NLI = {"hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large"}
ADV = {"citation_relocation", "distractor_parroting", "entity_swap", "hedge_insertion", "numeric_drift"}
FILES = ["results/preview_pilot/verdicts.jsonl", "results/citation_relocation_pilot/verdicts.jsonl"]


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # marginal probs
    pa1 = sum(a) / n
    pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def main():
    cell = defaultdict(dict)  # (seed, op) -> {judge: 1 faithful / 0 unfaithful}
    for fp in FILES:
        for line in Path(fp).open():
            r = json.loads(line)
            if r["operator"] not in ADV or r["judge"] not in JUDGES:
                continue
            if r.get("metadata", {}).get("error") or r["verdict"] not in ("faithful", "unfaithful"):
                continue
            cell[(r["seed_id"], r["operator"])][r["judge"]] = 1 if r["verdict"] == "faithful" else 0

    full = {k: v for k, v in cell.items() if len(v) == 7}
    print(f"Fully-scored adversarial cells (all 7 judges): {len(full)}")

    cols = {j: [v[j] for v in full.values()] for j in JUDGES}
    K = {}
    for x, y in combinations(JUDGES, 2):
        K[(x, y)] = K[(y, x)] = cohen_kappa(cols[x], cols[y])

    short = {"claude_sonnet_4_6": "Claude", "faithjudge_style_sonnet": "FaithJ",
             "mistral_large_2": "Mistral", "hhem_2_1_open": "HHEM",
             "minicheck_flan_t5_large": "MiniCk", "alignscore_large": "Align",
             "ragas_style_sonnet": "RAGAS"}

    lines = []
    lines.append(f"Pairwise Cohen's kappa, ADVERSARIAL pool only ({len(full)} cells, 5 operators, paraphrase_null excluded)")
    lines.append("=" * 92)
    hdr = "           " + " ".join(f"{short[j]:>8}" for j in JUDGES)
    lines.append(hdr)
    for x in JUDGES:
        row = f"{short[x]:>10} "
        for y in JUDGES:
            row += f"{'   --   ' if x == y else f'{K[(x,y)]:+.3f} ':>9}"
        lines.append(row)

    within_llm = [K[p] for p in combinations(sorted(LLM), 2)]
    within_nli = [K[p] for p in combinations(sorted(NLI), 2)]
    cross = [K[(x, y)] for x in LLM for y in NLI]
    lines.append("")
    lines.append(f"within-LLM avg kappa = {sum(within_llm)/len(within_llm):+.3f}")
    for (x, y) in combinations(sorted(LLM), 2):
        lines.append(f"    {short[x]} x {short[y]} = {K[(x,y)]:+.3f}")
    lines.append(f"within-NLI avg kappa = {sum(within_nli)/len(within_nli):+.3f}")
    for (x, y) in combinations(sorted(NLI), 2):
        lines.append(f"    {short[x]} x {short[y]} = {K[(x,y)]:+.3f}")
    lines.append(f"cross-architecture (LLM x NLI) avg kappa = {sum(cross)/len(cross):+.3f}")
    lines.append(f"within-LLM minus within-NLI gap = {sum(within_llm)/len(within_llm) - sum(within_nli)/len(within_nli):+.3f}")

    out = "\n".join(lines)
    print(out)
    Path("results/kappa_adversarial_pool.txt").write_text(out + "\n")
    print("\nWrote results/kappa_adversarial_pool.txt")


if __name__ == "__main__":
    main()
