"""Tests for the BYO-judge audit path. No API keys required."""

from __future__ import annotations

import re

import pytest

from jfre import audit_judge, make_seed, relocate_citations


def _seed():
    return make_seed(
        question="When did each event occur?",
        passages=[
            "The first satellite launched in 1957.",
            "The first crewed Moon landing was in 1969.",
            "The first reusable rocket landing was in 2015.",
        ],
        cited_answer=(
            "The first satellite launched in 1957 [1]. The first crewed Moon "
            "landing was in 1969 [2]. The first reusable rocket landing was in 2015 [3]."
        ),
        seed_id="space-0",
    )


def test_relocate_changes_citations_but_not_content():
    seed = _seed()
    clean = seed.metadata["cited_answer"]
    attacked = relocate_citations(seed)
    assert attacked != clean
    # The prose (everything outside the [N] markers) is unchanged.
    assert re.sub(r"\[\d+\]", "", attacked) == re.sub(r"\[\d+\]", "", clean)
    # Same multiset of citation indices, just permuted with no fixed point.
    assert sorted(re.findall(r"\[(\d+)\]", attacked)) == sorted(
        re.findall(r"\[(\d+)\]", clean)
    )
    for a, b in zip(re.findall(r"\[(\d+)\]", clean), re.findall(r"\[(\d+)\]", attacked)):
        assert a != b  # every marker moved


def test_content_only_judge_has_blind_spot():
    """A judge that only checks content support passes the relocated answer."""

    def content_only_judge(question, passages, answer):
        haystack = " ".join(passages).lower()
        claim = re.sub(r"\[\d+\]", "", answer).lower()
        words = [w for w in re.findall(r"[a-z0-9]+", claim) if len(w) > 3]
        return all(w in haystack for w in words)

    result = audit_judge(content_only_judge, [_seed()])
    assert result.n_attackable == 1
    assert result.clean_pass == 1  # passes the correctly-cited answer
    assert result.attacked_pass == 1  # ALSO passes the misattributed answer = miss
    assert result.false_negative_rate == 1.0


def test_attribution_aware_judge_catches_it():
    """A judge that checks each [N] against its cited passage flips to unfaithful."""

    def attribution_judge(question, passages, answer):
        # passages[i] supports a claim iff they share a salient token; check that
        # the claim segment ending in [N] is supported by passage N specifically.
        for segment in re.split(r"(?<=\])", answer):
            m = re.search(r"\[(\d+)\]", segment)
            if not m:
                continue
            n = int(m.group(1))
            cited = passages[n - 1].lower()
            claim = re.sub(r"\[\d+\]", "", segment).lower()
            salient = [w for w in re.findall(r"\d{4}|[a-z]{5,}", claim)]
            if not all(w in cited for w in salient):
                return False
        return True

    result = audit_judge(attribution_judge, [_seed()])
    assert result.clean_pass == 1
    assert result.attacked_pass == 0  # catches the misattribution
    assert result.false_negative_rate == 0.0
    assert result.catches_attribution == 1


def test_seed_with_too_few_citations_is_skipped():
    seed = make_seed(
        question="What is the capital?",
        passages=["The capital of France is Paris."],
        cited_answer="The capital of France is Paris [1].",
    )
    result = audit_judge(lambda q, p, a: True, [seed])
    assert result.n_attackable == 0
    assert result.n_skipped == 1
    assert "No attackable seeds" in result.summary()


def test_relocate_raises_when_not_attackable():
    seed = make_seed(
        question="q",
        passages=["only one passage [1]."],
        cited_answer="single claim [1].",
    )
    with pytest.raises(ValueError):
        relocate_citations(seed)
