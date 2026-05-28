"""Smoke test for AlignScore-large."""

from __future__ import annotations

from jfre.judges.alignscore_judge import score
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-alignscore-001",
    source="expertqa",
    question="When was Acme Robotics founded?",
    passages=[
        Passage(text="Acme Robotics was founded in 2017 by Maria Chen and David Park in Seattle.", is_relevant=True),
    ],
    gold_answer="Acme Robotics was founded in 2017 by Maria Chen and David Park.",
)


def main() -> None:
    print("Loading AlignScore-large (one-time ~4.7 GB checkpoint load)...")
    print()
    print("--- Clean ---")
    v = score(SEED, SEED.gold_answer, operator="clean")
    print(f"  verdict: {v.verdict}  raw_score: {v.raw_score}  err: {v.judge_metadata.get('error', 'none')}")

    perturbed = "Acme Robotics was founded in 2015 by Sarah Johnson and Michael Brown."
    print(f"\n--- Perturbed: {perturbed!r} ---")
    v = score(SEED, perturbed, operator="entity_swap")
    print(f"  verdict: {v.verdict}  raw_score: {v.raw_score}  err: {v.judge_metadata.get('error', 'none')}")


if __name__ == "__main__":
    main()
