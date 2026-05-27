"""Smoke test: numeric_drift operator on a hand-crafted fictional seed.

Run from repo root:
    python scripts/smoke_numeric_drift.py
"""

from __future__ import annotations

from jfre.operators.numeric_drift import generate
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-002",
    source="expertqa",
    question="When was Acme Robotics founded and how much did it raise in Series B?",
    passages=[
        Passage(
            text=(
                "Acme Robotics was founded in 2017 by Maria Chen and David Park in "
                "Seattle, with an initial focus on warehouse automation."
            ),
            is_relevant=True,
        ),
        Passage(
            text=(
                "In 2020, Acme Robotics raised a $40M Series B led by Sequoia Capital, "
                "doubling its engineering headcount."
            ),
            is_relevant=True,
        ),
    ],
    gold_answer="Acme Robotics was founded in 2017 and raised $40M in its Series B in 2020.",
)


def main() -> None:
    print(f"Seed: {SEED.seed_id}")
    print(f"  Q:    {SEED.question}")
    print(f"  Gold: {SEED.gold_answer}")
    print()

    p = generate(SEED)

    print(f"Operator:      {p.operator}")
    print(f"Rule passed:   {p.rule_passed}")
    print(f"Rule notes:    {p.rule_notes}")
    print(f"Edit diff:     {p.edit_diff}")
    print(f"Perturbed:     {p.perturbed_answer!r}")


if __name__ == "__main__":
    main()
