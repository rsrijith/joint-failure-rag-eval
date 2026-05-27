"""Smoke test for hedge_insertion on a few HotpotQA seeds."""

from __future__ import annotations

from jfre.operators.hedge_insertion import generate
from jfre.seeds.hotpotqa import load


def main() -> None:
    for seed in load(n=3, shuffle_seed=11):
        print(f"\n=== {seed.seed_id[:35]} ===")
        print(f"  Q:    {seed.question[:100]}")
        print(f"  Gold: {seed.gold_answer}")

        p = generate(seed)
        tag = "PASS" if p.rule_passed else "FAIL"
        print(f"  [{tag}] hedge_insertion: {p.rule_notes}")
        if p.rule_passed:
            print(f"     hedge:        {p.edit_diff['hedge_used']}")
            print(f"     added_claim:  {p.edit_diff['added_claim']}")
            print(f"     perturbed:    {p.perturbed_answer!r}")


if __name__ == "__main__":
    main()
