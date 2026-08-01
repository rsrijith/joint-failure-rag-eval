"""The drop-in fix for LLM-as-judge citation-attribution blindness.

For LLM judges the blind spot is a prompt gap, not a model limit. The deployed
prompt asks "is every claim supported by the passages" and never mentions
citation correctness. Adding one rule -- check that each [N] marker points to a
passage that actually supports the claim it is attached to -- cuts the
attribution miss rate from about 46% to about 3% in the reference study, while
still passing most correctly-cited answers.

Copy ``ATTRIBUTION_AWARE_PROMPT`` into your judge, or wrap your own LLM call:

    from jfre.fix import make_attribution_judge

    def call_my_llm(prompt: str) -> str:
        ...  # return the model's raw text response

    judge = make_attribution_judge(call_my_llm)   # -> a jfre Judge callable
    judge(question, passages, answer)             # True == faithful

For NLI judges (HHEM, MiniCheck, AlignScore) the gap is structural: they pool
the passages into one premise, so "the passage this claim cites" is not
represented. Scoring per (claim, cited-passage) pair recovers sensitivity but
over-rejects correct citations; no shipped NLI wrapper exposes that mode. The
prompt fix below applies to LLM-as-judge and claim-decomposition judges.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Sequence

# The attribution-aware faithfulness prompt. Identical content-support definition
# as a standard faithfulness judge, PLUS the explicit per-citation rule. This is
# the only change from a content-only prompt.
ATTRIBUTION_AWARE_PROMPT = """You are a faithfulness judge for a retrieval-augmented question answering system.

You are given a question, a set of numbered retrieved passages, and a candidate answer that contains [N] citation markers (N is the passage number being cited).

Decide whether the candidate answer is FAITHFUL or UNFAITHFUL.

Definitions:
- FAITHFUL: every factual claim is supported by the passages, AND every citation marker [N] points to a passage that actually supports the specific claim it is attached to.
- UNFAITHFUL: at least one claim is unsupported, contradicts a passage, OR carries a citation [N] where passage N does NOT support that claim -- even if some other passage would support it (this is citation misattribution).

Check each [N] marker: read passage N and confirm it supports the claim the marker is attached to. Judge relative to the passages only, not world knowledge.

Question:
{question}

Retrieved passages:
{passages}

Candidate answer:
{answer}

Respond with valid JSON only, no markdown fences:
{{"verdict": "faithful" OR "unfaithful", "reasoning": "one sentence"}}"""


def format_passages(passages: Sequence[str]) -> str:
    """Render passages with [PASSAGE n] tags so [N] markers resolve."""
    return "\n\n".join(f"[PASSAGE {i + 1}] {p}" for i, p in enumerate(passages))


def render_attribution_prompt(
    question: str, passages: Sequence[str], answer: str
) -> str:
    """Fill the attribution-aware prompt for one (question, passages, answer)."""
    return ATTRIBUTION_AWARE_PROMPT.format(
        question=question,
        passages=format_passages(passages),
        answer=answer,
    )


def parse_verdict(raw: str) -> bool | None:
    """Parse a judge response into faithful (True) / unfaithful (False) / None.

    None means the response could not be parsed as a faithful/unfaithful verdict.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        verdict = str(json.loads(s).get("verdict", "")).strip().lower()
    except (json.JSONDecodeError, AttributeError):
        return None
    if verdict == "faithful":
        return True
    if verdict == "unfaithful":
        return False
    return None


def make_attribution_judge(
    call_llm: Callable[[str], str],
    on_parse_error: bool = True,
) -> Callable[[str, list[str], str], bool]:
    """Wrap an LLM text-completion callable into an attribution-aware Judge.

    ``call_llm`` takes a prompt string and returns the model's raw text. The
    returned judge renders the attribution-aware prompt, calls the model, and
    parses the verdict. ``on_parse_error`` is the verdict used when the response
    cannot be parsed (default True == faithful, i.e. fail-open, matching the
    reference judges so a parse failure does not inflate the catch rate).
    """

    def judge(question: str, passages: list[str], answer: str) -> bool:
        raw = call_llm(render_attribution_prompt(question, passages, answer))
        verdict = parse_verdict(raw)
        return on_parse_error if verdict is None else verdict

    return judge
