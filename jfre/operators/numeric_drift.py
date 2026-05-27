"""Numeric-drift operator.

Perturb exactly one numeric value (year, percentage, count, amount) in the
gold answer to a plausibly-close but unsupported alternative. All non-numeric
tokens must be byte-identical. See METHODOLOGY.md §4.2.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import anthropic

from jfre.types import OperatorName, Passage, Perturbation, Seed


OPERATOR_NAME: OperatorName = "numeric_drift"

_GENERATION_MODEL = "claude-sonnet-4-6"

# Matches integers, decimals, percentages, comma-separated thousands.
# Boundaries use (?<![\w.]) and (?![\w.]) to avoid splitting tokens like "v2.1".
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    r"(?![\w.])"
)

_PROMPT = """You will perturb a faithful answer to make it unfaithful, by changing exactly ONE numeric value.

Question:
{question}

Retrieved context (passages, separated by [PASSAGE BREAK]):
{passages}

Faithful answer (gold human-written, do not change anything else in this text):
{gold_answer}

Your task:
1. Find ONE numeric value in the faithful answer. Acceptable kinds: year, count, percentage, amount, decimal, ordinal-with-digit.
2. Replace it with a PLAUSIBLY-CLOSE but incorrect alternative of the same type. Year for year, percent for percent, integer count for integer count, etc.
3. The new value must NOT appear in the retrieved context as supporting evidence for this claim.
4. Apply ONLY this one substitution. Do not change any other words, punctuation, capitalization, or whitespace.

Return JSON only, no markdown fences, no commentary:
{{"original_number": "...", "perturbed_number": "...", "perturbed_answer": "..."}}
"""


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _format_passages(passages: list[Passage]) -> str:
    return "\n[PASSAGE BREAK]\n".join(p.text for p in passages)


def _strip_json_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _mask_numbers(text: str) -> tuple[str, list[str]]:
    """Replace every numeric token with __NUM__ and return (masked text, numbers in order)."""
    numbers: list[str] = []

    def _capture(m: re.Match) -> str:
        numbers.append(m.group(0))
        return "__NUM__"

    masked = _NUMBER_RE.sub(_capture, text)
    return masked, numbers


def _validate(
    seed: Seed,
    original_number: str,
    perturbed_number: str,
    perturbed_answer: str,
) -> tuple[bool, str]:
    """Automated rule for numeric-drift (METHODOLOGY.md §4.2).

    Pass iff:
      - When numeric tokens are masked, gold and perturbed answers are byte-identical
        (i.e., only numbers changed, no other tokens edited).
      - Exactly one numeric token changed.
      - The perturbed number is not a verbatim substring of any retrieved passage.
      - The original and perturbed numbers are distinct.
    """
    gold_masked, gold_nums = _mask_numbers(seed.gold_answer)
    perturbed_masked, perturbed_nums = _mask_numbers(perturbed_answer)

    if gold_masked != perturbed_masked:
        return False, "non-numeric tokens differ between gold and perturbed answer"
    if len(gold_nums) != len(perturbed_nums):
        return False, f"number count differs (gold={len(gold_nums)}, perturbed={len(perturbed_nums)})"

    diffs = [i for i, (g, p) in enumerate(zip(gold_nums, perturbed_nums)) if g != p]
    if len(diffs) == 0:
        return False, "no numeric change detected"
    if len(diffs) != 1:
        return False, f"expected exactly 1 numeric change, found {len(diffs)}"

    actual_original = gold_nums[diffs[0]]
    actual_perturbed = perturbed_nums[diffs[0]]

    if actual_original != original_number:
        return False, f"model claimed original={original_number!r} but actual diff is {actual_original!r}"
    if actual_perturbed != perturbed_number:
        return False, f"model claimed perturbed={perturbed_number!r} but actual diff is {actual_perturbed!r}"
    if actual_original == actual_perturbed:
        return False, "original and perturbed numbers are identical"

    # Check the perturbed number does not appear in any retrieved passage
    # (using \b boundaries via a regex to avoid matching e.g. "20" inside "2020").
    perturbed_re = re.compile(rf"(?<![\w.]){re.escape(actual_perturbed)}(?![\w.])")
    for passage in seed.passages:
        if perturbed_re.search(passage.text):
            return False, f"perturbed number {actual_perturbed!r} appears in passage; not unsupported"

    return True, "ok"


def generate(seed: Seed) -> Perturbation:
    """Generate a numeric-drift perturbation for one seed."""
    prompt = _PROMPT.format(
        question=seed.question,
        passages=_format_passages(seed.passages),
        gold_answer=seed.gold_answer,
    )

    msg = _client().messages.create(
        model=_GENERATION_MODEL,
        max_tokens=1024,
        system="You are a careful annotator. Respond with valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    raw = _strip_json_fences(raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            perturbed_answer="",
            edit_diff={"raw_model_output": raw},
            rule_passed=False,
            rule_notes=f"JSON parse failure: {e}",
        )

    required = {"original_number", "perturbed_number", "perturbed_answer"}
    missing = required - parsed.keys()
    if missing:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            perturbed_answer="",
            edit_diff={"raw_model_output": raw, "parsed": parsed},
            rule_passed=False,
            rule_notes=f"missing fields: {sorted(missing)}",
        )

    passed, notes = _validate(
        seed,
        original_number=parsed["original_number"],
        perturbed_number=parsed["perturbed_number"],
        perturbed_answer=parsed["perturbed_answer"],
    )

    return Perturbation(
        seed_id=seed.seed_id,
        operator=OPERATOR_NAME,
        perturbed_answer=parsed["perturbed_answer"] if passed else "",
        edit_diff={
            "original_number": parsed["original_number"],
            "perturbed_number": parsed["perturbed_number"],
        },
        rule_passed=passed,
        rule_notes=notes,
    )
