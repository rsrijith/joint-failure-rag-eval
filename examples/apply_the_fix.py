"""Apply the attribution-aware fix to an LLM-as-judge. No API keys required to read.

The fix is one prompt change. This example shows the before/after wiring; plug in
your real model where ``call_llm`` is defined.

    python examples/apply_the_fix.py
"""

from __future__ import annotations

from jfre.fix import (
    ATTRIBUTION_AWARE_PROMPT,
    make_attribution_judge,
    render_attribution_prompt,
)


def call_llm(prompt: str) -> str:
    """Replace with your model. Must return the model's raw text response.

    Example (Anthropic):
        import anthropic
        msg = anthropic.Anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=256, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")
    """
    raise NotImplementedError("plug in your model here")


def main() -> None:
    question = "When did each event occur?"
    passages = [
        "The first satellite launched in 1957.",
        "The first crewed Moon landing was in 1969.",
    ]
    answer = "The first satellite launched in 1957 [1]. The first Moon landing was in 1969 [2]."

    print("The drop-in prompt (also in FIX.md):\n")
    print(ATTRIBUTION_AWARE_PROMPT)
    print("\n--- Rendered for one example ---\n")
    print(render_attribution_prompt(question, passages, answer))

    # In your code, this single line upgrades a content-only judge to an
    # attribution-aware one:
    judge = make_attribution_judge(call_llm)  # noqa: F841
    print(
        "\nmake_attribution_judge(call_llm) returns a jfre Judge you can pass to "
        "audit_judge(...), or call directly as judge(question, passages, answer)."
    )


if __name__ == "__main__":
    main()
