"""Claude 4 LLM-judge (Anthropic API, claude-opus-4-7)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import anthropic

from jfre.judges._llm_judge_prompt import parse_verdict, render_prompt
from jfre.judges._retry import retry_on_rate_limit
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "claude_opus_4_7"
_MODEL = "claude-opus-4-7"


@lru_cache(maxsize=1)
def _client() -> anthropic.Anthropic:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=key)


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    """Score the (seed.question, seed.passages, answer_to_judge) tuple.

    operator is recorded on the verdict for downstream stratification
    ("clean" for the unperturbed gold answer, otherwise the operator name).
    """
    prompt = render_prompt(seed.question, seed.passages, answer_to_judge)

    # claude-opus-4-7 does not accept temperature; rely on API default for determinism.
    @retry_on_rate_limit()
    def _call():
        return _client().messages.create(
            model=_MODEL,
            max_tokens=512,
            system="You are a careful faithfulness judge. Respond with valid JSON only.",
            messages=[{"role": "user", "content": prompt}],
        )

    msg = _call()
    raw = "".join(b.text for b in msg.content if b.type == "text")

    verdict, reasoning, debug = parse_verdict(raw)

    if verdict == "parse_error":
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",  # default-fail-open: don't bias the joint-failure toward unfaithful
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
