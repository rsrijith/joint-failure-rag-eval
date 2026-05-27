"""Entity-swap operator.

Replace exactly one named entity in the gold answer with a plausible
same-class alternative that is not supported by the retrieved context.
See METHODOLOGY.md §4.1.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import anthropic

from jfre.types import OperatorName, Passage, Perturbation, Seed


OPERATOR_NAME: OperatorName = "entity_swap"

# Generation model. Strong enough to follow the single-substitution constraint;
# distinct from the LLM-as-judge models used for evaluation.
_GENERATION_MODEL = "claude-sonnet-4-6"

_PROMPT = """You will perturb a faithful answer to make it unfaithful, by swapping exactly ONE named entity for a plausible same-class alternative that is NOT supported by the retrieved context.

Question:
{question}

Retrieved context (passages, separated by [PASSAGE BREAK]):
{passages}

Faithful answer (gold human-written, do not change anything else in this text):
{gold_answer}

Your task:
1. Find ONE named entity in the faithful answer. Acceptable NER classes: PERSON, ORG, LOC, GPE, PRODUCT, EVENT, DATE, NUMBER.
2. Replace it with a PLAUSIBLE same-class alternative that is NOT mentioned anywhere in the retrieved context as supporting this claim.
3. Apply ONLY this one substitution. Do not change any other words, punctuation, or formatting.

Return JSON only, no markdown fences, no commentary:
{{"ner_class": "...", "original_entity": "...", "perturbed_entity": "...", "perturbed_answer": "..."}}
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


def _validate(
    seed: Seed,
    original_entity: str,
    perturbed_entity: str,
    perturbed_answer: str,
) -> tuple[bool, str]:
    """Automated rule for entity-swap (METHODOLOGY.md §4.1).

    Pass iff:
      - original_entity appears exactly once in the gold answer
      - perturbed_answer equals gold_answer with exactly that substring replaced
        (no other tokens changed)
      - perturbed_entity does not appear verbatim in any retrieved passage
        (the swap is genuinely unsupported)
      - original_entity != perturbed_entity (something actually changed)
    """
    gold = seed.gold_answer

    if original_entity not in gold:
        return False, f"original_entity {original_entity!r} not in gold answer"
    count = gold.count(original_entity)
    if count != 1:
        return False, f"original_entity {original_entity!r} appears {count} times in gold; ambiguous"

    expected_perturbed = gold.replace(original_entity, perturbed_entity, 1)
    if expected_perturbed != perturbed_answer:
        return False, "perturbed_answer is not gold_answer with exactly that substring replaced"

    for passage in seed.passages:
        if perturbed_entity in passage.text:
            return False, f"perturbed_entity {perturbed_entity!r} appears in passage; not unsupported"

    if original_entity.strip().lower() == perturbed_entity.strip().lower():
        return False, "original_entity and perturbed_entity are identical"

    return True, "ok"


def generate(seed: Seed) -> Perturbation:
    """Generate an entity-swap perturbation for one seed.

    Returns a Perturbation. If the automated rule fails, perturbed_answer is
    empty and rule_passed is False; the caller should discard.
    """
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

    required = {"ner_class", "original_entity", "perturbed_entity", "perturbed_answer"}
    missing = required - parsed.keys()
    if missing:
        return Perturbation(
            seed_id=seed.seed_id,
            operator=OPERATOR_NAME,
            perturbed_answer="",
            edit_diff={"raw_model_output": raw, "parsed": parsed},
            rule_passed=False,
            rule_notes=f"missing fields in model output: {sorted(missing)}",
        )

    passed, notes = _validate(
        seed,
        original_entity=parsed["original_entity"],
        perturbed_entity=parsed["perturbed_entity"],
        perturbed_answer=parsed["perturbed_answer"],
    )

    return Perturbation(
        seed_id=seed.seed_id,
        operator=OPERATOR_NAME,
        perturbed_answer=parsed["perturbed_answer"] if passed else "",
        edit_diff={
            "ner_class": parsed["ner_class"],
            "original_entity": parsed["original_entity"],
            "perturbed_entity": parsed["perturbed_entity"],
        },
        rule_passed=passed,
        rule_notes=notes,
    )
