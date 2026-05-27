"""HotpotQA seed loader.

Reads the canonical dev-distractor JSON from data/raw/hotpotqa/
(run scripts/download_hotpotqa.py first to fetch it).

HotpotQA distractor split: 10 passages per question — 2 supporting +
8 distractors. Maps to our (multi-passage context, gold answer) seed
format. License: CC-BY-SA 4.0.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path

from jfre.types import Passage, Seed


_LOCAL = Path("data/raw/hotpotqa/hotpot_dev_distractor_v1.json")
_YES_NO = {"yes", "no"}


def _has_perturbable_content(answer: str) -> bool:
    """Filter out yes/no, very short, and content-poor answers.

    Operators need an entity or a numeric token to grab onto; short/boolean
    answers are not useful seeds for the pilot.
    """
    if answer.strip().lower() in _YES_NO:
        return False
    if len(answer.split()) < 3:
        return False
    has_entity_candidate = bool(re.search(r"\b[A-Z][a-z]+", answer))
    has_number = bool(re.search(r"\d", answer))
    return has_entity_candidate or has_number


def load(n: int, shuffle_seed: int = 42) -> Iterator[Seed]:
    """Yield up to N HotpotQA seeds that pass the perturbable-content filter."""
    if not _LOCAL.exists():
        raise FileNotFoundError(
            f"{_LOCAL} not found. Run: python scripts/download_hotpotqa.py"
        )

    with _LOCAL.open() as f:
        examples = json.load(f)

    rng = random.Random(shuffle_seed)
    rng.shuffle(examples)

    yielded = 0
    for example in examples:
        if yielded >= n:
            return
        answer = example["answer"]
        if not _has_perturbable_content(answer):
            continue

        # supporting_facts: list of [title, sent_idx] pairs.
        supporting_titles = {sf[0] for sf in example["supporting_facts"]}

        # context: list of [title, [sentences]] pairs (10 items in distractor split).
        passages: list[Passage] = []
        for title, sentences in example["context"]:
            text = " ".join(sentences).strip()
            passages.append(Passage(text=text, is_relevant=(title in supporting_titles)))

        yield Seed(
            seed_id=f"hotpotqa-{example['_id']}",
            source="hotpotqa",
            question=example["question"].strip(),
            passages=passages,
            gold_answer=answer.strip(),
            metadata={
                "hop_type": example.get("type", "unknown"),   # "comparison" | "bridge"
                "level": example.get("level", "unknown"),     # "easy" | "medium" | "hard"
            },
        )
        yielded += 1
