"""Claude 4 LLM-judge (Anthropic API, claude-opus-4-7)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import anthropic

from jfre.judges._llm_judge_prompt import parse_verdict, render_prompt
from jfre.judges._retry import retry_on_rate_limit
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "claude_sonnet_4_6"
_MODEL = "claude-sonnet-4-6"
# Note: earlier pilot runs used claude-opus-4-7 (JUDGE_NAME="claude_opus_4_7").
# Those verdicts remain in verdicts.jsonl as a separate judge. Switching to
# Sonnet + prompt caching cuts Claude API cost by ~15x for binary faithful/
# unfaithful classification with negligible quality loss.


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

    @retry_on_rate_limit()
    def _call():
        return _client().messages.create(
            model=_MODEL,
            max_tokens=512,
            temperature=0,
            # System prompt is identical across every judge call, so cache it.
            # 5-min ephemeral cache; hits cost 10% of base input price.
            system=[{
                "type": "text",
                "text": "You are a careful faithfulness judge. Respond with valid JSON only.",
                "cache_control": {"type": "ephemeral"},
            }],
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
