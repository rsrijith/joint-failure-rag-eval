"""RAGAS-style faithfulness judge implemented directly via Claude Sonnet.

The official `ragas` package fails on Python 3.14 due to an asyncio/anyio
incompatibility ('NoneType' object has no attribute 'set_name'). We
implement the same methodology directly:

Step 1: decompose the answer into atomic factual claims (1 LLM call).
Step 2: for each claim, verify whether it is supported by the retrieved
        context (one LLM call per claim).
Step 3: faithfulness score = (# supported claims) / (# total claims).

This matches RAGAS faithfulness as documented in Es et al. (EACL 2024 Demos)
and the ragas source.
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


JUDGE_NAME = "ragas_style_sonnet"
_MODEL = "claude-sonnet-4-6"
_THRESHOLD = 0.5

_DECOMPOSE_PROMPT = """Decompose the following answer into a list of atomic factual claims. Each claim should be a single fact that could be independently verified against a source document.

Answer:
{answer}

Return ONLY a JSON list of strings, no markdown fences, no commentary. Example: ["claim 1", "claim 2", "claim 3"]"""


_VERIFY_PROMPT = """Determine whether the following CLAIM is supported by the CONTEXT below.

CONTEXT:
{context}

CLAIM:
{claim}

A claim is "supported" if all of the factual content in it can be directly inferred from the context. A claim is "unsupported" if it contains any factual content that is not in or contradicts the context.

Return ONLY a JSON object, no markdown fences:
{{"supported": true OR false}}"""


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


@retry_on_rate_limit()
def _decompose(answer: str) -> list[str]:
    msg = _client().messages.create(
        model=_MODEL,
        max_tokens=1024,
        temperature=0,
        system=[{
            "type": "text",
            "text": "You decompose answers into atomic factual claims. Respond with valid JSON only.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": _DECOMPOSE_PROMPT.format(answer=answer)}],
    )
    raw = _strip_fences("".join(b.text for b in msg.content if b.type == "text"))
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"expected list, got {type(parsed).__name__}")
    return [str(c).strip() for c in parsed if str(c).strip()]


@retry_on_rate_limit()
def _verify(context: str, claim: str) -> bool:
    # Cache the context block: same context is reused across N claims per seed,
    # and across N×6 calls per seed (clean + 5 perturbed). Cache hits give 10x
    # cost reduction on the context portion.
    msg = _client().messages.create(
        model=_MODEL,
        max_tokens=128,
        temperature=0,
        system=[{
            "type": "text",
            "text": "You verify whether a claim is supported by a context. Respond with valid JSON only.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"CONTEXT:\n{context}",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": f"\nCLAIM:\n{claim}\n\nReturn ONLY a JSON object, no markdown fences:\n{{\"supported\": true OR false}}"},
            ],
        }],
    )
    raw = _strip_fences("".join(b.text for b in msg.content if b.type == "text"))
    parsed = json.loads(raw)
    return bool(parsed.get("supported", False))


def score(
    seed: Seed,
    answer_to_judge: str,
    operator: OperatorName | Literal["clean"],
) -> JudgeVerdict:
    """Score one (seed, candidate answer) via RAGAS-style claim decomposition."""
    context = format_passages(seed.passages)

    try:
        claims = _decompose(answer_to_judge)
    except Exception as e:
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",  # fail-open
            raw_score=None,
            judge_metadata={"error": f"decompose failed: {e}", "model": _MODEL},
        )

    if not claims:
        return JudgeVerdict(
            seed_id=seed.seed_id,
            operator=operator,
            judge_name=JUDGE_NAME,
            verdict="faithful",
            raw_score=1.0,
            judge_metadata={"n_claims": 0, "n_supported": 0, "model": _MODEL},
        )

    n_supported = 0
    verify_errors = 0
    for claim in claims:
        try:
            if _verify(context, claim):
                n_supported += 1
        except Exception:
            verify_errors += 1

    raw_score = n_supported / len(claims)
    verdict = "faithful" if raw_score >= _THRESHOLD else "unfaithful"

    return JudgeVerdict(
        seed_id=seed.seed_id,
        operator=operator,
        judge_name=JUDGE_NAME,
        verdict=verdict,
        raw_score=raw_score,
        judge_metadata={
            "threshold": _THRESHOLD,
            "score": raw_score,
            "n_claims": len(claims),
            "n_supported": n_supported,
            "verify_errors": verify_errors,
            "model": _MODEL,
        },
    )
