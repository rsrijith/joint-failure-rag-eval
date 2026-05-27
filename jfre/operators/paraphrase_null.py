"""Faithfulness-preserving syntactic paraphrase (negative control).

Rephrase the gold answer with different syntax while preserving meaning.
A faithful gold should still be faithful after paraphrasing; any judge that
flips on this operator has a robustness problem orthogonal to faithfulness.

See METHODOLOGY.md §4.5.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import anthropic

from jfre.types import OperatorName, Perturbation, Seed


OPERATOR_NAME: OperatorName = "paraphrase_null"

_GENERATION_MODEL = "claude-sonnet-4-6"
_EQUIVALENCE_MODEL = "claude-sonnet-4-6"

_PARAPHRASE_PROMPT = """Rephrase the following answer using different syntax (e.g., switch between active and passive voice, reorder clauses, change word order) WHILE PRESERVING THE EXACT SAME MEANING.

Constraints:
- Do NOT add any new information.
- Do NOT remove any information.
- Do NOT change any factual content (names, numbers, dates, places).
- Only change syntax/wording style.
- The result must be a complete, grammatical answer to the question.

Question:
{question}

Faithful answer (gold):
{gold_answer}

Return JSON only, no markdown fences, no commentary:
{{"paraphrased_answer": "..."}}
"""

_EQUIVALENCE_PROMPT = """Decide whether the two answers below convey the exact same factual content. Same meaning even if the wording differs is "yes"; any added, removed, or changed factual content is "no".

Answer A: {a}

Answer B: {b}

Return JSON only, no markdown fences:
{{"equivalent": true OR false, "reasoning": "one sentence"}}
"""


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _strip_json_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _check_equivalence(a: str, b: str) -> tuple[bool, str]:
    """Ask Claude whether two answers convey the same content."""
    prompt = _EQUIVALENCE_PROMPT.format(a=a, b=b)
    msg = _client().messages.create(
        model=_EQUIVALENCE_MODEL,
        max_tokens=256,
        system="You are a careful annotator. Respond with valid JSON only.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text")
    raw = _strip_json_fences(raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"equivalence-check JSON parse failure: {raw[:120]}"
    equivalent = bool(parsed.get("equivalent", False))
    reasoning = str(parsed.get("reasoning", ""))
    return equivalent, reasoning


def _validate(seed: Seed, paraphrased_answer: str) -> tuple[bool, str]:
    """Automated rule for paraphrase_null (METHODOLOGY.md §4.5).

    Pass iff:
      - paraphrased_answer is not byte-identical to gold (something actually changed)
      - length is within 0.5x-2x of gold (sanity check against degeneration)
      - separate Claude equivalence call returns "yes"
    """
    if paraphrased_answer.strip() == seed.gold_answer.strip():
        return False, "paraphrased answer is identical to gold (no perturbation)"

    pa_len = len(paraphrased_answer.split())
    gold_len = len(seed.gold_answer.split())
    if pa_len < 0.5 * gold_len or pa_len > 2.0 * gold_len:
        return False, f"length out of band (gold={gold_len} words, paraphrase={pa_len})"

    equivalent, reasoning = _check_equivalence(seed.gold_answer, paraphrased_answer)
    if not equivalent:
        return False, f"equivalence check failed: {reasoning[:120]}"

    return True, f"ok ({reasoning[:80]})"


def generate(seed: Seed) -> Perturbation:
    """Generate a paraphrase_null perturbation (negative control)."""
    prompt = _PARAPHRASE_PROMPT.format(
        question=seed.question,
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

    paraphrased = parsed.get("paraphrased_answer", "")
    if not paraphrased:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            perturbed_answer="",
            edit_diff={"parsed": parsed},
            rule_passed=False,
            rule_notes="missing paraphrased_answer field",
        )

    passed, notes = _validate(seed, paraphrased)

    return Perturbation(
        seed_id=seed.seed_id,
        operator=OPERATOR_NAME,
        perturbed_answer=paraphrased if passed else "",
        edit_diff={"paraphrased_answer": paraphrased},
        rule_passed=passed,
        rule_notes=notes,
    )
