"""GLM-4.7 LLM-judge via Cerebras inference API.

Cerebras's free tier model menu has churned: originally Llama-3.3-70B
(never available), substituted with Qwen3-235B which Cerebras retired
between pilot iterations. GLM-4.7 (Z.AI, Chinese frontier model) is the
remaining frontier-tier open-weights option. Qwen verdicts from earlier
runs remain in verdicts.jsonl under the "qwen3_235b_cerebras" name; new
calls record under "glm_4_7_cerebras".
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Literal

from cerebras.cloud.sdk import Cerebras

from jfre.judges._llm_judge_prompt import parse_verdict, render_prompt
from jfre.judges._retry import retry_on_rate_limit
from jfre.types import JudgeVerdict, OperatorName, Seed


JUDGE_NAME = "glm_4_7_cerebras"
_MODEL = "zai-glm-4.7"

# Self-imposed rate limit. Cerebras free tier "queue_exceeded" errors fire
# when too many concurrent requests are queued, so we pace ourselves to stay
# below ~40 requests/minute. Sleep is taken BEFORE each call.
_INTER_REQUEST_DELAY_S = 1.5
_last_call_time = 0.0

# Module-level flag set after the first token_quota_exceeded. Cerebras's free
# tier resets daily, so once exhausted we short-circuit subsequent calls for
# the rest of the process lifetime to avoid wasting 1.5 s of throttle per call.
_quota_dead = False
_QUOTA_DEAD_MARKERS = ("tokens per day", "token_quota_exceeded", "quota exceeded")


@lru_cache(maxsize=1)
def _client() -> Cerebras:
    key = os.environ.get("CEREBRAS_API_KEY")
    if not key:
        raise RuntimeError("CEREBRAS_API_KEY not set")
    return Cerebras(api_key=key)


def _throttle() -> None:
    """Sleep just enough to keep the call rate below the self-imposed limit."""
    global _last_call_time
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < _INTER_REQUEST_DELAY_S:
        time.sleep(_INTER_REQUEST_DELAY_S - elapsed)
    _last_call_time = time.monotonic()


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    global _quota_dead
    if _quota_dead:
        raise RuntimeError("glm_cerebras quota_dead (skipping for rest of session)")

    prompt = render_prompt(seed.question, seed.passages, answer_to_judge)

    _throttle()

    # GLM-4.7 is a reasoning model that emits chain-of-thought in a separate
    # `reasoning` field but still consumes max_tokens for it. 512 truncates
    # before content is produced; bump to 2048 to leave room.
    @retry_on_rate_limit()
    def _call():
        return _client().chat.completions.create(
            model=_MODEL,
            max_tokens=2048,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a careful faithfulness judge. Respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )

    try:
        response = _call()
    except Exception as e:
        if any(m in str(e).lower() for m in _QUOTA_DEAD_MARKERS):
            _quota_dead = True
        raise
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
