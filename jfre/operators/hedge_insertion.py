"""Hedge-insertion operator.

Append ONE hedged, unsupported claim to the gold answer. The original
gold answer is preserved verbatim; the perturbation is purely additive.

Targets LLM-as-judge architectures that may interpret hedged language as
"speculative content is permissible" and rate the perturbed answer faithful.
See METHODOLOGY.md §4.3.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

import anthropic

from jfre.types import OperatorName, Passage, Perturbation, Seed


OPERATOR_NAME: OperatorName = "hedge_insertion"

_GENERATION_MODEL = "claude-sonnet-4-6"

# Approved hedges. The perturbation must use one of these (lowercase comparison).
HEDGE_LEXICON: tuple[str, ...] = (
    "possibly",
    "presumably",
    "reportedly",
    "allegedly",
    "supposedly",
    "according to some sources",
    "it has been reported that",
    "it is said that",
    "some sources suggest",
    "rumored to",
    "speculation suggests",
)


_PROMPT = """You will perturb a faithful answer by APPENDING ONE additional hedged claim that is NOT supported by the retrieved context.

Question:
{question}

Retrieved context (passages, separated by [PASSAGE BREAK]):
{passages}

Faithful answer (gold human-written — do NOT modify this text; only append):
{gold_answer}

Your task:
1. Compose ONE new factual claim related to the topic that is NOT supported anywhere in the retrieved context.
2. Prepend a HEDGE PHRASE from this approved list: {hedge_lexicon}.
3. Append the hedged claim to the END of the faithful answer, separated by a space.

Constraints:
- Preserve the faithful answer character-for-character; do not edit it.
- Append ONLY the hedged unsupported claim, nothing else.
- The added claim must be plausible but NOT verifiable from the retrieved context.

Return JSON only, no markdown fences, no commentary:
{{"hedge_used": "...", "added_claim": "...", "perturbed_answer": "..."}}
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
    hedge_used: str,
    added_claim: str,
    perturbed_answer: str,
) -> tuple[bool, str]:
    """Automated rule for hedge_insertion (METHODOLOGY.md §4.3).

    Pass iff:
      - perturbed answer contains the full gold answer verbatim (preserved)
      - perturbed answer is strictly longer than the gold answer
      - some hedge from HEDGE_LEXICON appears in the appended portion
      - the added_claim text actually appears in the perturbed answer
      - the added_claim is not a verbatim substring of any retrieved passage
    """
    gold = seed.gold_answer

    if gold not in perturbed_answer:
        return False, "gold answer not preserved verbatim in perturbed answer"

    if len(perturbed_answer) <= len(gold):
        return False, "perturbed answer is not strictly longer than gold (nothing added)"

    if added_claim and added_claim not in perturbed_answer:
        return False, f"added_claim {added_claim[:40]!r} not present in perturbed answer"

    hedge_lc = hedge_used.lower().strip()
    hedge_match = any(h in hedge_lc or hedge_lc in h for h in HEDGE_LEXICON)
    if not hedge_match:
        return False, f"hedge_used {hedge_used!r} not in approved lexicon"

    appended_substring = perturbed_answer.split(gold, 1)[-1]
    if hedge_lc not in appended_substring.lower():
        return False, "hedge phrase not found in appended portion of perturbed answer"

    for passage in seed.passages:
        if added_claim and added_claim in passage.text:
            return False, "added_claim appears in a retrieved passage; not unsupported"

    return True, "ok"


def generate(seed: Seed) -> Perturbation:
    """Generate a hedge_insertion perturbation for one seed."""
    prompt = _PROMPT.format(
        question=seed.question,
        passages=_format_passages(seed.passages),
        gold_answer=seed.gold_answer,
        hedge_lexicon=", ".join(f'"{h}"' for h in HEDGE_LEXICON),
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

    required = {"hedge_used", "added_claim", "perturbed_answer"}
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
        hedge_used=parsed["hedge_used"],
        added_claim=parsed["added_claim"],
        perturbed_answer=parsed["perturbed_answer"],
    )

    return Perturbation(
        seed_id=seed.seed_id,
        operator=OPERATOR_NAME,
        perturbed_answer=parsed["perturbed_answer"] if passed else "",
        edit_diff={
            "hedge_used": parsed["hedge_used"],
            "added_claim": parsed["added_claim"],
        },
        rule_passed=passed,
        rule_notes=notes,
    )
