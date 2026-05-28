"""Smoke test for FaithJudge-style judge."""

from __future__ import annotations

from jfre.judges.faithjudge import score
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-faithjudge-001",
    source="expertqa",
    question="When was Acme Robotics founded and by whom?",
    passages=[
        Passage(
            text="Acme Robotics was founded in 2017 by Maria Chen and David Park in Seattle.",
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
    print()
    print("--- Clean ---")
    v = score(SEED, SEED.gold_answer, operator="clean")
    print(f"  verdict: {v.verdict}  type: {v.judge_metadata.get('hallucination_type')}  reason: {v.judge_metadata.get('reasoning', '')[:100]}")

    perturbed = "Acme Robotics was founded in 2015 by Sarah Johnson and Michael Brown."
    print(f"\n--- Perturbed: {perturbed!r} ---")
    v = score(SEED, perturbed, operator="entity_swap")
    print(f"  verdict: {v.verdict}  type: {v.judge_metadata.get('hallucination_type')}  reason: {v.judge_metadata.get('reasoning', '')[:100]}")


if __name__ == "__main__":
    main()
