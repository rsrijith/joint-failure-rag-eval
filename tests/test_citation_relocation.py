"""Tests for the citation_relocation operator. No API keys required."""

from __future__ import annotations

import re

from jfre.operators import citation_relocation
from jfre.types import Passage, Seed


def _cited_seed(cited_answer: str) -> Seed:
    return Seed(
        seed_id="t0",
        source="expertqa",
        question="q",
        passages=[Passage(text=f"p{i}", is_relevant=True) for i in range(1, 5)],
        gold_answer=cited_answer,
        metadata={"cited_answer": cited_answer},
    )


def test_derangement_has_no_fixed_points():
    cited = "a [1]. b [2]. c [3]. d [4]."
    pert = citation_relocation.generate(_cited_seed(cited))
    assert pert.rule_passed
    before = re.findall(r"\[(\d+)\]", cited)
    after = re.findall(r"\[(\d+)\]", pert.perturbed_answer)
    assert sorted(before) == sorted(after)
    assert all(a != b for a, b in zip(before, after))


def test_deterministic_per_seed():
    cited = "a [1]. b [2]. c [3]."
    p1 = citation_relocation.generate(_cited_seed(cited))
    p2 = citation_relocation.generate(_cited_seed(cited))
    assert p1.perturbed_answer == p2.perturbed_answer  # seeded RNG


def test_requires_cited_answer():
    seed = Seed(
        seed_id="t1",
        source="expertqa",
        question="q",
        passages=[Passage(text="p", is_relevant=True)],
        gold_answer="no citations here",
        metadata={},
    )
    pert = citation_relocation.generate(seed)
    assert not pert.rule_passed
    assert "no cited_answer" in pert.rule_notes


def test_requires_two_distinct_indices():
    pert = citation_relocation.generate(_cited_seed("a [1]. b [1]."))
    assert not pert.rule_passed
    assert "need >= 2" in pert.rule_notes
