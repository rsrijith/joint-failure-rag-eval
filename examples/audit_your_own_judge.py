"""Audit your own faithfulness judge for the citation-attribution blind spot.

Runs with NO API keys: the two judges below are illustrative stand-ins so you can
see the mechanism. Replace ``content_only_judge`` with your real RAGAS / HHEM /
LLM-as-judge / custom metric and run again.

    python examples/audit_your_own_judge.py
"""

from __future__ import annotations

import re

from jfre import audit_judge, make_seed
from jfre.fix import make_attribution_judge

# --- Your seeds: (question, passages, cited answer with [N] markers) -----------
# Each passage is numbered from 1; [N] in the answer cites passage N.
SEEDS = [
    make_seed(
        question="When did each event occur?",
        passages=[
            "The first satellite launched in 1957.",
            "The first crewed Moon landing was in 1969.",
            "The first reusable rocket landing was in 2015.",
        ],
        cited_answer=(
            "The first satellite launched in 1957 [1]. The first crewed Moon "
            "landing was in 1969 [2]. The first reusable rocket landing was in "
            "2015 [3]."
        ),
        seed_id="space-0",
    ),
    make_seed(
        question="What are the capitals?",
        passages=[
            "The capital of France is Paris.",
            "The capital of Japan is Tokyo.",
        ],
        cited_answer="The capital of France is Paris [1]. The capital of Japan is Tokyo [2].",
        seed_id="capitals-0",
    ),
]


# --- A content-only judge (the common case): passes if every word of the answer
#     appears somewhere in the passages, ignoring which passage is cited. This is
#     the blind spot the audit exposes. ---------------------------------------
def content_only_judge(question: str, passages: list[str], answer: str) -> bool:
    haystack = " ".join(passages).lower()
    claim = re.sub(r"\[\d+\]", "", answer).lower()
    words = [w for w in re.findall(r"[a-z0-9]+", claim) if len(w) > 3]
    return all(w in haystack for w in words)


# --- An attribution-aware judge built from a toy "LLM" that actually reads the
#     cited passage. In practice you pass your real model to make_attribution_judge.
def toy_attribution_llm(prompt: str) -> str:
    """A stand-in for a real model call, so this example runs with no API keys.

    It does what the attribution-aware prompt asks: for each ``[N]`` marker, read
    passage N and check that it supports the claim the marker is attached to. A
    real model replaces this entirely; the point is that the wiring and the
    verdict format are identical.
    """
    import json

    # The prompt renders passages as "[PASSAGE n] ..." and then the answer.
    passages = dict(
        (int(n), text.strip())
        for n, text in re.findall(r"\[PASSAGE (\d+)\]\s*(.*?)(?=\n\n\[PASSAGE |\n\nCandidate answer:)", prompt, re.S)
    )
    m = re.search(r"Candidate answer:\n(.*?)\n\nRespond with valid JSON", prompt, re.S)
    answer = m.group(1) if m else ""

    # Split the answer into (claim, cited passage number) pairs.
    claims = re.findall(r"([^.]*?)\[(\d+)\]", answer)
    if not claims:
        return json.dumps({"verdict": "faithful", "reasoning": "no citations to check"})

    def overlap(claim_words: list[str], passage: str) -> int:
        low = passage.lower()
        return sum(1 for w in claim_words if w in low)

    for claim, n in claims:
        n = int(n)
        words = [w for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) > 3]
        if not words:
            continue
        # Which passage actually supports this claim best? A correctly-cited claim
        # should match its own passage at least as well as any other. Comparing
        # against all passages is what makes this robust to boilerplate words the
        # passages share (e.g. "capital"), which a raw threshold gets wrong.
        scores = {i: overlap(words, text) for i, text in passages.items()}
        best = max(scores.values(), default=0)
        if scores.get(n, 0) < best:
            return json.dumps(
                {
                    "verdict": "unfaithful",
                    "reasoning": f"claim cites [{n}] but passage {max(scores, key=scores.get)} supports it better",
                }
            )
    return json.dumps({"verdict": "faithful", "reasoning": "every citation resolves to a supporting passage"})


def main() -> None:
    print("=== Your content-only judge ===")
    print(audit_judge(content_only_judge, SEEDS).summary())

    print("\n=== With the attribution-aware fix (make_attribution_judge) ===")
    fixed = make_attribution_judge(toy_attribution_llm)
    print(audit_judge(fixed, SEEDS).summary())

    print(
        "\nReplace content_only_judge with your real metric, and toy_attribution_llm\n"
        "with a real model call, to measure (and close) the blind spot on your stack."
    )


if __name__ == "__main__":
    main()
