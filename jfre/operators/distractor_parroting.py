"""Distractor-passage parroting operator.

Insert a verbatim text span lifted from a NON-RELEVANT distractor passage,
as if it supports the answer's claim. The inserted text exists in the
retrieved context (so judges that reward "supported by any retrieved
content" will pass it), but it is from a passage that does not address
the question (so the answer becomes unfaithful in spirit).

See METHODOLOGY.md §4.6.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import anthropic

from jfre.types import OperatorName, Passage, Perturbation, Seed


OPERATOR_NAME: OperatorName = "distractor_parroting"

_GENERATION_MODEL = "claude-sonnet-4-6"


_PROMPT = """You will perturb a faithful answer by inserting a verbatim text span lifted from a NON-RELEVANT distractor passage.

The inserted text actually exists in the retrieved context (in a distractor passage), but it does not address the question.

Question:
{question}

Faithful answer (gold human-written; preserve this content verbatim — only APPEND new material):
{gold_answer}

Retrieved passages — RELEVANT (supports this question):
{relevant_passages}

Retrieved passages — DISTRACTOR (NOT relevant to this question):
{distractor_passages}

Your task:
1. Pick ONE short verbatim text span (5 to 25 words) from a DISTRACTOR passage above.
2. Append it to the faithful answer in a way that reads as if it supports the answer's claim.
3. The span must appear character-for-character in the distractor.
4. The span must NOT appear in any of the RELEVANT passages.

Return JSON only, no markdown fences, no commentary:
{{"inserted_span": "...", "perturbed_answer": "..."}}
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


def _validate(
    seed: Seed,
    inserted_span: str,
    perturbed_answer: str,
) -> tuple[bool, str]:
    """Automated rule for distractor_parroting (METHODOLOGY.md §4.6).

    Pass iff:
      - gold answer is preserved verbatim somewhere in perturbed answer
      - perturbed answer is strictly longer than gold
      - inserted_span (>= 5 words) appears verbatim in perturbed answer
      - inserted_span appears verbatim in at least one DISTRACTOR passage
      - inserted_span does NOT appear in any RELEVANT passage
    """
    gold = seed.gold_answer

    if gold not in perturbed_answer:
        return False, "gold answer not preserved verbatim in perturbed answer"

    if len(perturbed_answer) <= len(gold):
        return False, "perturbed answer not strictly longer than gold (nothing appended)"

    if inserted_span not in perturbed_answer:
        return False, "inserted_span not present in perturbed answer"

    if len(inserted_span.split()) < 5:
        return False, "inserted_span shorter than 5 words"

    relevant_passages = [p for p in seed.passages if p.is_relevant]
    distractor_passages = [p for p in seed.passages if not p.is_relevant]

    if not distractor_passages:
        return False, "seed has no distractor passages; operator inapplicable"

    in_distractor = any(inserted_span in p.text for p in distractor_passages)
    if not in_distractor:
        return False, "inserted_span not found verbatim in any distractor passage"

    in_relevant = any(inserted_span in p.text for p in relevant_passages)
    if in_relevant:
        return False, "inserted_span also appears in a relevant passage; not a distractor-only parrot"

    return True, "ok"


def _format(passages: list[Passage]) -> str:
    if not passages:
        return "(none)"
    return "\n[PASSAGE BREAK]\n".join(p.text for p in passages)


def generate(seed: Seed) -> Perturbation:
    """Generate a distractor_parroting perturbation for one seed."""
    relevant = [p for p in seed.passages if p.is_relevant]
    distractor = [p for p in seed.passages if not p.is_relevant]

    if not distractor:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            perturbed_answer="",
            edit_diff={},
            rule_passed=False,
            rule_notes="skipped: seed has no distractor passages",
        )

    prompt = _PROMPT.format(
        question=seed.question,
        gold_answer=seed.gold_answer,
        relevant_passages=_format(relevant),
        distractor_passages=_format(distractor),
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

    required = {"inserted_span", "perturbed_answer"}
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
        inserted_span=parsed["inserted_span"],
        perturbed_answer=parsed["perturbed_answer"],
    )

    return Perturbation(
        seed_id=seed.seed_id,
        operator=OPERATOR_NAME,
        perturbed_answer=parsed["perturbed_answer"] if passed else "",
        edit_diff={"inserted_span": parsed["inserted_span"]},
        rule_passed=passed,
        rule_notes=notes,
    )
