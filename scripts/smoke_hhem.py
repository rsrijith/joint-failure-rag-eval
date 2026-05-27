"""Smoke test for HHEM-2.1-Open. Downloads the model on first run (~700 MB).

Run from repo root:
    python scripts/smoke_hhem.py
"""

from __future__ import annotations

from jfre.judges.hhem_judge import score
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-hhem-001",
    source="expertqa",
    question="When was Acme Robotics founded and by whom?",
    passages=[
        Passage(
            text=(
                "Acme Robotics was founded in 2017 by Maria Chen and David Park in Seattle, "
                "with an initial focus on warehouse automation."
            ),
            is_relevant=True,
        ),
        Passage(
            text=(
                "In 2020, Acme Robotics raised a $40M Series B led by Sequoia Capital."
            ),
            is_relevant=False,
        ),
    ],
    gold_answer="Acme Robotics was founded in 2017 by Maria Chen and David Park.",
)


def main() -> None:
    print("Loading HHEM model (first call downloads weights)...")

    print("\n  Clean (faithful) answer:")
    v = score(SEED, SEED.gold_answer, operator="clean")
    print(f"    verdict: {v.verdict}  (raw score: {v.raw_score:.3f})")

    perturbed = "Acme Robotics was founded in 2015 by Sarah Johnson and Michael Brown."
    print(f"\n  Perturbed (unfaithful): {perturbed!r}")
    v = score(SEED, perturbed, operator="entity_swap")
    print(f"    verdict: {v.verdict}  (raw score: {v.raw_score:.3f})")


if __name__ == "__main__":
    main()
