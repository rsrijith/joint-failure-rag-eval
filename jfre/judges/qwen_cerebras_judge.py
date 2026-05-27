"""Qwen3-235B LLM-judge via Cerebras inference API.

Originally planned as Llama-3.3-70B, substituted because Cerebras's free
tier does not host the 70B Llama. Qwen3-235B-A22B-Instruct-2507 is a
frontier-tier Mixture-of-Experts model from Alibaba — same role in the
ensemble (open-weights LLM-judge from a different organization than
Anthropic) and benchmarks at or above Llama-3.3-70B on most evaluations.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from cerebras.cloud.sdk import Cerebras

from jfre.judges._llm_judge_prompt import parse_verdict, render_prompt
from jfre.judges._retry import retry_on_rate_limit
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "qwen3_235b_cerebras"
_MODEL = "qwen-3-235b-a22b-instruct-2507"


@lru_cache(maxsize=1)
def _client() -> Cerebras:
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    return Cerebras(api_key=key)


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    prompt = render_prompt(seed.question, seed.passages, answer_to_judge)

    @retry_on_rate_limit()
    def _call():
        return _client().chat.completions.create(
            model=_MODEL,
            max_tokens=512,
            temperature=0,
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
