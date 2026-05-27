"""Smoke test for MiniCheck (downloads ~3 GB on first run)."""

from __future__ import annotations

from jfre.judges.minicheck_judge import score
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-minicheck-001",
    source="expertqa",
    question="When was Acme Robotics founded?",
    passages=[
        Passage(
            text="Acme Robotics was founded in 2017 by Maria Chen and David Park in Seattle.",
            is_relevant=True,
        ),
    ],
    gold_answer="Acme Robotics was founded in 2017 by Maria Chen and David Park.",
)


def main() -> None:
    print("Loading MiniCheck (first call may download ~3 GB)...")

    print("\n  Clean (faithful) answer:")
    v = score(SEED, SEED.gold_answer, operator="clean")
    print(f"    verdict: {v.verdict}  raw_score: {v.raw_score:.3f}  backend: {v.judge_metadata.get('backend')}")

    perturbed = "Acme Robotics was founded in 2015 by Maria Chen and David Park."
    print(f"\n  Perturbed (numeric drift): {perturbed!r}")
    v = score(SEED, perturbed, operator="numeric_drift")
    print(f"    verdict: {v.verdict}  raw_score: {v.raw_score:.3f}")


if __name__ == "__main__":
    main()
