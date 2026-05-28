"""Smoke test for ExpertQA loader."""

from __future__ import annotations

from jfre.seeds.expertqa import load


def main() -> None:
    seeds = list(load(n=3))
    print(f"Loaded {len(seeds)} ExpertQA seeds.\n")

    for seed in seeds:
        print(f"=== {seed.seed_id} ({seed.metadata.get('field', 'unknown')}) ===")
        print(f"  Q:    {seed.question[:120]}")
        print(f"  Gold: {seed.gold_answer[:200]}")
        print(f"  Passages: {len(seed.passages)} (all relevant; no distractors in ExpertQA)")
        if seed.passages:
            print(f"  First passage ({len(seed.passages[0].text)} chars): {seed.passages[0].text[:160]}...")
        print()


if __name__ == "__main__":
    main()
