"""Smoke test for paraphrase_null on 3 HotpotQA seeds."""

from __future__ import annotations

from jfre.operators.paraphrase_null import generate
from jfre.seeds.hotpotqa import load


def main() -> None:
    for seed in load(n=3, shuffle_seed=17):
        print(f"\n=== {seed.seed_id[:35]} ===")
        print(f"  Q:    {seed.question[:80]}")
        print(f"  Gold: {seed.gold_answer}")

        p = generate(seed)
        tag = "PASS" if p.rule_passed else "FAIL"
        print(f"  [{tag}] paraphrase_null: {p.rule_notes}")
        if p.rule_passed:
            print(f"     paraphrase: {p.perturbed_answer!r}")


if __name__ == "__main__":
    main()
