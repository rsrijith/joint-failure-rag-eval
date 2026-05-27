"""Mistral Large 2 LLM-judge (Mistral API)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from mistralai.client.sdk import Mistral

from jfre.judges._llm_judge_prompt import parse_verdict, render_prompt
from jfre.judges._retry import retry_on_rate_limit
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "mistral_large_2"
_MODEL = "mistral-large-latest"


@lru_cache(maxsize=1)
def _client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    return Mistral(api_key=key)


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    prompt = render_prompt(seed.question, seed.passages, answer_to_judge)

    @retry_on_rate_limit()
    def _call():
        return _client().chat.complete(
            model=_MODEL,
            temperature=0,
            max_tokens=512,
            messages=[
                {"role": "system", "content": "You are a careful faithfulness judge. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

    response = _call()
    raw = response.choices[0].message.content or ""

    verdict, reasoning, debug = parse_verdict(raw)

    if verdict == "parse_error":
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",
            raw_score=None,
            judge_metadata={"error": "parse_error", "reasoning": reasoning, **debug},
        )

    return JudgeVerdict(
        seed_id=seed.seed_id,
        operator=operator,
        judge_name=JUDGE_NAME,
        verdict=verdict,
        raw_score=None,
        judge_metadata={"reasoning": reasoning, "model": _MODEL},
    )
