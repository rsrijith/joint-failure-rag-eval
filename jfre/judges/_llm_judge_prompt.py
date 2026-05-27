"""Shared LLM-as-judge prompt + parsing helpers.

Used by claude_judge, llama_judge, mistral_judge — all three share the same
faithfulness rubric and JSON output format so verdicts are comparable.
"""

from __future__ import annotations

import json
import re

from jfre.types import Passage


PROMPT_TEMPLATE = """You are a faithfulness judge for a retrieval-augmented question answering system.

You are given a question, a set of retrieved passages, and a candidate answer.

Decide whether the candidate answer is FAITHFUL or UNFAITHFUL to the retrieved passages.

Definitions:
- FAITHFUL: every factual claim in the candidate answer is directly supported by content in the retrieved passages.
- UNFAITHFUL: the candidate answer contains at least one factual claim that is not supported by, or contradicts, any retrieved passage.

Do NOT use your own world knowledge to verify claims. Judge faithfulness only relative to the retrieved passages. A factually-true claim that is not in the passages is UNFAITHFUL for the purposes of this evaluation.

Question:
{question}

Retrieved passages:
{passages}

Candidate answer:
{answer}

Respond with valid JSON only, no markdown fences, no commentary:
{{"verdict": "faithful" OR "unfaithful", "reasoning": "one sentence"}}
"""


def format_passages(passages: list[Passage]) -> str:
    """Render passages with [PASSAGE n] tags so the judge can reference them."""
    return "\n\n".join(f"[PASSAGE {i + 1}] {p.text}" for i, p in enumerate(passages))


def render_prompt(question: str, passages: list[Passage], answer: str) -> str:
    return PROMPT_TEMPLATE.format(
        question=question,
        passages=format_passages(passages),
        answer=answer,
    )


def strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_verdict(raw: str) -> tuple[str, str, dict]:
    """Parse a judge's raw JSON response.

    Returns (verdict, reasoning, debug_info). verdict is "faithful",
    "unfaithful", or "parse_error". reasoning is the one-line justification
    or the parse error message.
    """
    cleaned = strip_fences(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return "parse_error", f"JSON parse: {e}", {"raw": raw}

    verdict_raw = str(parsed.get("verdict", "")).strip().lower()
    reasoning = str(parsed.get("reasoning", "")).strip()

    if verdict_raw not in {"faithful", "unfaithful"}:
        return "parse_error", f"unrecognized verdict: {verdict_raw!r}", {"parsed": parsed}

    return verdict_raw, reasoning, {"parsed": parsed}
