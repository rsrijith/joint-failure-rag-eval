"""Compare the attribution-prompt 2x2 on INDEPENDENT (Llama-annotated) citations
against the original Claude-annotated result. Tests whether the attribution Δ
survives an annotator that is not one of the judges (circularity control).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

LLM = ["claude_sonnet_4_6", "mistral_large_2", "faithjudge_style_sonnet"]
CROSS = Path("results/citation_relocation_pilot/verdicts_crossmodel_attribution.jsonl")

# Original Claude-annotated 2x2 faithful-rates (from attribution_prompt_result.md)
CLAUDE_ANNOT = {
    "claude_sonnet_4_6":      {"content_clean": 49, "content_scrambled": 49, "attr_clean": 85, "attr_scrambled": 3},
    "mistral_large_2":        {"content_clean": 50, "content_scrambled": 51, "attr_clean": 69, "attr_scrambled": 3},
    "faithjudge_style_sonnet":{"content_clean": 41, "content_scrambled": 37, "attr_clean": 79, "attr_scrambled": 2},
}


def main():
    # cross-model verdicts: judge -> (condition, prompt_type) -> [faithful, total]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    n_seeds = set()
    for line in CROSS.open():
        r = json.loads(line)
        if r["judge"] not in LLM or r["verdict"] not in ("faithful", "unfaithful"):
            continue
        n_seeds.add(r["seed_id"])
        cell = agg[r["judge"]][(r["condition"], r["prompt_type"])]
        cell[1] += 1
        if r["verdict"] == "faithful":
            cell[0] += 1

    def pct(judge, cond, ptype):
        f, t = agg[judge][(cond, ptype)]
        return (100 * f / t) if t else float("nan"), t

    print("=" * 86)
    print("  CIRCULARITY CONTROL: attribution 2x2 on Llama-3.3-70B-annotated citations")
    print(f"  (independent annotator, not a judge; {len(n_seeds)} usable seeds)")
    print("=" * 86)
    print("  faithful-rate %. content: clean≈scrambled => blind. attribution: clean HIGH, scrambled LOW => discriminates.\n")
    print(f"  {'judge':<24} {'content:clean':>13} {'content:scr':>12} {'attr:clean':>11} {'attr:scr':>9} {'attr Δ':>8}")
    print("  " + "-" * 82)
    for j in LLM:
        cc, _ = pct(j, "clean", "content")
        cs, _ = pct(j, "scrambled", "content")
        ac, _ = pct(j, "clean", "attribution")
        a_s, _ = pct(j, "scrambled", "attribution")
        delta = ac - a_s
        print(f"  {j:<24} {cc:>12.0f}% {cs:>11.0f}% {ac:>10.0f}% {a_s:>8.0f}% {delta:>+7.0f}pp")

    print("\n  Side-by-side attribution Δ (clean − scrambled faithful-rate):")
    print(f"  {'judge':<24} {'Claude-annotated':>18} {'Llama-annotated':>17}")
    for j in LLM:
        ca = CLAUDE_ANNOT[j]["attr_clean"] - CLAUDE_ANNOT[j]["attr_scrambled"]
        ac, _ = pct(j, "clean", "attribution")
        a_s, _ = pct(j, "scrambled", "attribution")
        print(f"  {j:<24} {ca:>16}pp {ac - a_s:>15.0f}pp")
    print("\n  If the Llama-annotated Δ is comparable to Claude-annotated, the")
    print("  annotator-judge circularity objection does not explain the effect.")


if __name__ == "__main__":
    main()
