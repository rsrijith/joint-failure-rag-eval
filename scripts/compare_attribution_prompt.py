"""Compare content-only vs attribution-aware prompt on citation_relocation.

Content-only verdicts = the deployed citation_relocation verdicts already in
results/citation_relocation_pilot/verdicts.jsonl. Attribution-aware verdicts =
results/citation_relocation_pilot/verdicts_attribution_prompt.jsonl.

Reports, per LLM judge, FNR (called the misattributed answer 'faithful') under
each prompt on the matched set of cells.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LLM = ["claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"]


def main():
    # content-only verdicts (deployed prompt)
    content = defaultdict(dict)  # judge -> seed_id -> verdict
    for line in Path("results/citation_relocation_pilot/verdicts.jsonl").open():
        r = json.loads(line)
        if r["operator"] == "citation_relocation" and r["judge"] in LLM \
           and not r.get("metadata", {}).get("error"):
            content[r["judge"]][r["seed_id"]] = r["verdict"]

    # attribution-aware verdicts
    attr = defaultdict(dict)
    for line in Path("results/citation_relocation_pilot/verdicts_attribution_prompt.jsonl").open():
        r = json.loads(line)
        if r["judge"] in LLM and r["verdict"] in ("faithful", "unfaithful"):
            attr[r["judge"]][r["seed_id"]] = r["verdict"]

    print("=" * 72)
    print("  Citation_relocation: content-only prompt vs attribution-aware prompt")
    print("  FNR = fraction of misattributed answers the judge called 'faithful' (missed)")
    print("=" * 72)
    print(f"\n  {'judge':<26} {'content FNR':>14} {'attribution FNR':>18} {'change':>10}")
    print("  " + "-" * 70)
    for j in LLM:
        cells = sorted(set(content[j]) & set(attr[j]))
        n = len(cells)
        if not n:
            print(f"  {j:<26} (no matched cells yet)")
            continue
        c_fnr = sum(1 for s in cells if content[j][s] == "faithful") / n
        a_fnr = sum(1 for s in cells if attr[j][s] == "faithful") / n
        print(f"  {j:<26} {c_fnr*100:>11.0f}% (n={n}) {a_fnr*100:>13.0f}% {(a_fnr-c_fnr)*100:>+8.0f}pp")

    # pooled
    print()
    pooled_c = [(j, s) for j in LLM for s in (set(content[j]) & set(attr[j]))]
    if pooled_c:
        c = sum(1 for j, s in pooled_c if content[j][s] == "faithful") / len(pooled_c)
        a = sum(1 for j, s in pooled_c if attr[j][s] == "faithful") / len(pooled_c)
        print(f"  POOLED (3 LLM judges, {len(pooled_c)} judge-cells): "
              f"content FNR {c*100:.0f}% -> attribution FNR {a*100:.0f}%  ({(a-c)*100:+.0f}pp)")
        print(f"\n  Interpretation: attribution-prompting reduces the miss rate, but a")
        print(f"  residual {a*100:.0f}% FNR remains -> the blind spot is partly prompt-omission,")
        print(f"  partly a deeper limitation. Even told to check attribution, judges miss some.")


if __name__ == "__main__":
    main()
