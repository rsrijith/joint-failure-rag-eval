"""FaithJudge-style judge: Claude Sonnet with curated hallucination few-shot.

Implements the FaithJudge methodology (Tamber, Bao et al., EMNLP 2025 Industry)
directly via Claude Sonnet. FaithJudge uses an LLM-as-judge with few-shot
examples of common hallucination types from FaithBench. The actual Vectara
prompt + few-shot pool is published in github.com/vectara/FaithJudge; we use
a representative subset here.

This is "FaithJudge-style" not the exact model. Documented in §8 limitations.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Literal

import anthropic

from jfre.judges._llm_judge_prompt import format_passages
from jfre.judges._retry import retry_on_rate_limit
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "faithjudge_style_sonnet"
_MODEL = "claude-sonnet-4-6"

# Few-shot examples drawn from common hallucination patterns documented in
# the FaithBench paper (Bao et al. 2025): unverified intrinsic, unverified
# extrinsic, intrinsic factual error, extrinsic factual error.
_FEW_SHOT = """Below are examples of how to judge faithfulness:

EXAMPLE 1 (faithful):
CONTEXT: The Eiffel Tower was completed in 1889 and stands 330 meters tall.
ANSWER: The Eiffel Tower, completed in 1889, is 330 meters in height.
VERDICT: faithful (every claim in the answer is directly supported)

EXAMPLE 2 (unfaithful — extrinsic error):
CONTEXT: The Eiffel Tower was completed in 1889 and stands 330 meters tall.
ANSWER: The Eiffel Tower was designed by Alexandre Gustave Eiffel in 1889.
VERDICT: unfaithful (the designer's name is not in the context — extrinsic hallucination)

EXAMPLE 3 (unfaithful — numeric drift):
CONTEXT: The Eiffel Tower was completed in 1889 and stands 330 meters tall.
ANSWER: The Eiffel Tower was completed in 1888 and stands 330 meters tall.
VERDICT: unfaithful (year 1888 contradicts the context's 1889)

EXAMPLE 4 (unfaithful — hedged but unsupported):
CONTEXT: The Eiffel Tower was completed in 1889 and stands 330 meters tall.
ANSWER: The Eiffel Tower was completed in 1889; some sources suggest it was nearly demolished in 1909.
VERDICT: unfaithful (the demolition claim, even hedged, is not in the context)

EXAMPLE 5 (unfaithful — distractor parroting):
CONTEXT: [PASSAGE 1] The Eiffel Tower was completed in 1889.
         [PASSAGE 2] The Burj Khalifa is 828 meters tall, located in Dubai.
ANSWER: The Eiffel Tower was completed in 1889 and is 828 meters tall.
VERDICT: unfaithful (828m comes from passage about Burj Khalifa, not Eiffel)
"""


_PROMPT = """You are a strict faithfulness judge for retrieval-augmented question answering.

{few_shot}

Now evaluate the following:

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Apply the same standard as the examples above. Return JSON only, no markdown fences:
{{"verdict": "faithful" OR "unfaithful", "hallucination_type": "none" OR one of: ["unverified_extrinsic", "intrinsic_contradiction", "numeric_drift", "distractor_parroting", "hedged_unsupported", "other"], "reasoning": "one sentence"}}"""


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    prompt = _PROMPT.format(
        few_shot=_FEW_SHOT,
        context=format_passages(seed.passages),
        question=seed.question,
        answer=answer_to_judge,
    )

    @retry_on_rate_limit()
    def _call():
        return _client().messages.create(
            model=_MODEL,
            max_tokens=512,
            temperature=0,
            system="You are a strict faithfulness judge. Respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )

    try:
        msg = _call()
    except Exception as e:
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",
            raw_score=None,
            judge_metadata={"error": str(e)[:200], "model": _MODEL},
        )

    raw = _strip_fences("".join(b.text for b in msg.content if b.type == "text"))
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",
            raw_score=None,
            judge_metadata={"error": "parse_error", "raw": raw[:200], "model": _MODEL},
        )

    verdict_raw = str(parsed.get("verdict", "")).strip().lower()
    if verdict_raw not in {"faithful", "unfaithful"}:
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",
            raw_score=None,
            judge_metadata={"error": f"bad verdict: {verdict_raw}", "model": _MODEL},
        )

    return JudgeVerdict(
        seed_id=seed.seed_id,
        operator=operator,
        judge_name=JUDGE_NAME,
        verdict=verdict_raw,
        raw_score=None,
        judge_metadata={
            "reasoning": str(parsed.get("reasoning", "")),
            "hallucination_type": str(parsed.get("hallucination_type", "none")),
            "model": _MODEL,
        },
    )
