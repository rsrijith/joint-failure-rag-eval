"""Tests for the attribution-aware fix helpers. No API keys required."""

from __future__ import annotations

import json

from jfre.fix import (
    ATTRIBUTION_AWARE_PROMPT,
    make_attribution_judge,
    parse_verdict,
    render_attribution_prompt,
)


def test_prompt_mentions_per_citation_check():
    # The defining property of the fix: it asks about the SPECIFIC cited passage.
    assert "citation marker [N]" in ATTRIBUTION_AWARE_PROMPT
    assert "even if some other passage would support it" in ATTRIBUTION_AWARE_PROMPT


def test_render_includes_passages_and_answer():
    prompt = render_attribution_prompt(
        "Q?", ["alpha passage", "beta passage"], "claim [2]."
    )
    assert "[PASSAGE 1] alpha passage" in prompt
    assert "[PASSAGE 2] beta passage" in prompt
    assert "claim [2]." in prompt
    assert "Q?" in prompt


def test_parse_verdict_handles_plain_and_fenced_json():
    assert parse_verdict('{"verdict": "faithful", "reasoning": "x"}') is True
    assert parse_verdict('{"verdict": "unfaithful"}') is False
    assert parse_verdict('```json\n{"verdict": "faithful"}\n```') is True
    assert parse_verdict("not json") is None
    assert parse_verdict('{"verdict": "maybe"}') is None


def test_make_attribution_judge_wires_llm_call():
    seen = {}

    def fake_llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return json.dumps({"verdict": "unfaithful", "reasoning": "misattributed"})

    judge = make_attribution_judge(fake_llm)
    assert judge("Q?", ["p1", "p2"], "claim [1].") is False
    assert "[PASSAGE 1] p1" in seen["prompt"]


def test_parse_error_is_fail_open_by_default():
    judge = make_attribution_judge(lambda p: "garbage")
    assert judge("Q?", ["p1", "p2"], "claim [1].") is True  # fail-open default
    strict = make_attribution_judge(lambda p: "garbage", on_parse_error=False)
    assert strict("Q?", ["p1", "p2"], "claim [1].") is False
