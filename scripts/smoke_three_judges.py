"""Smoke test: 3 API judges score a (clean, perturbed) answer pair.

Verifies that all 3 judges respond and parse, and that they tend to call
clean answers faithful and entity-swapped answers unfaithful.

Run from repo root:
    python scripts/smoke_three_judges.py
"""

from __future__ import annotations

from jfre.judges import claude_judge, mistral_judge, qwen_cerebras_judge
from jfre.operators.entity_swap import generate as entity_swap_generate
from jfre.seeds.hotpotqa import load


JUDGES = [
    ("claude",  claude_judge),
    ("qwen",    qwen_cerebras_judge),
    ("mistral", mistral_judge),
]


def main() -> None:
    seeds = list(load(n=2, shuffle_seed=7))

    for seed in seeds:
        print(f"\n=== {seed.seed_id} ===")
        print(f"  Q:    {seed.question}")
        print(f"  Gold: {seed.gold_answer}")

        # Score the clean (unperturbed) gold answer first
        print("\n  Clean (gold) answer — should be FAITHFUL:")
        for name, mod in JUDGES:
            v = mod.score(seed, seed.gold_answer, operator="clean")
            print(f"    {name:8s} -> {v.verdict:11s}  ({v.judge_metadata.get('reasoning', '')[:90]})")

        # Apply entity_swap and score the perturbed answer
        p = entity_swap_generate(seed)
        if not p.rule_passed:
            print(f"\n  entity_swap failed: {p.rule_notes}")
            continue

        print(f"\n  Perturbed (entity_swap): {p.perturbed_answer!r}")
        print(f"  Edit: {p.edit_diff}")
        print("  Each judge should ideally call this UNFAITHFUL (perturbation broke faithfulness):")
        for name, mod in JUDGES:
            v = mod.score(seed, p.perturbed_answer, operator="entity_swap")
            print(f"    {name:8s} -> {v.verdict:11s}  ({v.judge_metadata.get('reasoning', '')[:90]})")


if __name__ == "__main__":
    main()
