"""End-to-end smoke test: load 3 HotpotQA seeds, run both operators on each.

Run from repo root:
    python scripts/smoke_hotpotqa_with_operators.py
"""

from __future__ import annotations

from jfre.operators import entity_swap, numeric_drift
from jfre.seeds.hotpotqa import load


def main() -> None:
    seeds = list(load(n=3))
    print(f"Loaded {len(seeds)} HotpotQA seeds.\n")

    for seed in seeds:
        print(f"=== {seed.seed_id} ({seed.metadata['hop_type']}, {seed.metadata['level']}) ===")
        print(f"  Q:    {seed.question}")
        print(f"  Gold: {seed.gold_answer}")
        print(f"  Passages: {len(seed.passages)} ({sum(p.is_relevant for p in seed.passages)} relevant)")
        print()

        for op_name, op_module in [("entity_swap", entity_swap), ("numeric_drift", numeric_drift)]:
            p = op_module.generate(seed)
            tag = "PASS" if p.rule_passed else "FAIL"
            print(f"  [{tag}] {op_name}")
            print(f"        notes: {p.rule_notes}")
            if p.rule_passed:
                print(f"        edit:  {p.edit_diff}")
                print(f"        perturbed: {p.perturbed_answer!r}")
            print()


if __name__ == "__main__":
    main()
