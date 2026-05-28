"""Smoke test for RAGAS judge.

Tests one clean (should be faithful) and one perturbed (should be unfaithful).
Prints raw faithfulness score for each.
"""

from __future__ import annotations

from jfre.judges.ragas_judge import score
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-ragas-001",
    source="expertqa",
    question="When was Acme Robotics founded and by whom?",
    passages=[
        Passage(
            text="Acme Robotics was founded in 2017 by Maria Chen and David Park in Seattle, focusing on warehouse automation.",
            is_relevant=True,
        ),
        Passage(
            text="In 2020, Acme Robotics raised a $40M Series B led by Sequoia Capital.",
            is_relevant=False,
        ),
    ],
    gold_answer="Acme Robotics was founded in 2017 by Maria Chen and David Park.",
)


def main() -> None:
    print(f"Seed: {SEED.seed_id}")
    print(f"Q: {SEED.question}")
    print(f"Gold: {SEED.gold_answer}")
    print()

    print("--- Clean (faithful) ---")
    v = score(SEED, SEED.gold_answer, operator="clean")
    print(f"  verdict: {v.verdict}  raw_score: {v.raw_score}  err: {v.judge_metadata.get('error', 'none')}")

    perturbed = "Acme Robotics was founded in 2015 by Sarah Johnson and Michael Brown."
    print(f"\n--- Perturbed: {perturbed!r} ---")
    v = score(SEED, perturbed, operator="entity_swap")
    print(f"  verdict: {v.verdict}  raw_score: {v.raw_score}  err: {v.judge_metadata.get('error', 'none')}")


if __name__ == "__main__":
    main()
