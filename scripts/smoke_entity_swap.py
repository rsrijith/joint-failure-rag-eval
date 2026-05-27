"""Smoke test: run entity_swap on a hand-crafted fictional seed.

Run from the repo root with the venv activated:
    python scripts/smoke_entity_swap.py
"""

from __future__ import annotations

from jfre.operators.entity_swap import generate
from jfre.types import Passage, Seed


SEED = Seed(
    seed_id="smoke-001",
    source="expertqa",
    question="When was Acme Robotics founded and by whom?",
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
            is_relevant=False,
        ),
    ],
    gold_answer="Acme Robotics was founded in 2017 by Maria Chen and David Park.",
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
