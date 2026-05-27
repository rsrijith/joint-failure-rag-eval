"""Shared types for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SeedSource = Literal["hotpotqa", "expertqa"]
OperatorName = Literal[
    "entity_swap",
    "numeric_drift",
    "hedge_insertion",
    "citation_relocation",
    "paraphrase_null",
    "distractor_parroting",
]
Verdict = Literal["faithful", "unfaithful"]


@dataclass
class Passage:
    """A single retrieved passage."""

    text: str
    is_relevant: bool  # True for gold-supporting, False for distractor


@dataclass
class Seed:
    """A faithful (question, multi-passage context, gold answer) tuple."""

    seed_id: str
    source: SeedSource
    question: str
    passages: list[Passage]
    gold_answer: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Perturbation:
    """A perturbed answer derived from a seed."""

    seed_id: str
    operator: OperatorName
    perturbed_answer: str
    edit_diff: dict  # operator-specific record of what changed
    rule_passed: bool  # did the operator-specific automated rule accept this?
    rule_notes: str = ""  # diagnostic info from the rule check


@dataclass
class JudgeVerdict:
    """One judge's verdict on one (perturbed) answer."""

    seed_id: str
    operator: OperatorName | Literal["clean"]  # "clean" = unperturbed gold
    judge_name: str
    verdict: Verdict
    raw_score: float | None  # for judges that return a continuous score
    judge_metadata: dict = field(default_factory=dict)
