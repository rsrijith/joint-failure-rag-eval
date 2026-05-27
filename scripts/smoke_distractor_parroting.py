"""Smoke test for distractor_parroting on 3 HotpotQA seeds."""

from __future__ import annotations

from jfre.operators.distractor_parroting import generate
from jfre.seeds.hotpotqa import load


def main() -> None:
    for seed in load(n=3, shuffle_seed=13):
        print(f"\n=== {seed.seed_id[:35]} ===")
        n_dist = sum(1 for p in seed.passages if not p.is_relevant)
        n_rel = sum(1 for p in seed.passages if p.is_relevant)
        print(f"  Q:    {seed.question[:80]}")
        print(f"  Gold: {seed.gold_answer}")
        print(f"  Passages: {n_rel} relevant, {n_dist} distractor")

        p = generate(seed)
        tag = "PASS" if p.rule_passed else "FAIL"
        print(f"  [{tag}] distractor_parroting: {p.rule_notes}")
        if p.rule_passed:
            print(f"     inserted span: {p.edit_diff['inserted_span']!r}")
            print(f"     perturbed:     {p.perturbed_answer!r}")


if __name__ == "__main__":
    main()
