"""Attribution-aware prompt experiment on citation_relocation.

The deployed judge prompt asks "is every claim supported by the passages" and
never mentions citation correctness. This experiment re-scores all 100
citation_relocation perturbations with an ATTRIBUTION-AWARE prompt for each of
the 3 LLM-as-judge judges (Claude, Mistral, FaithJudge), then compares to their
existing content-only verdicts.

Intervention is the prompt only; same model, same answer, same passages. NLI
judges (HHEM/MiniCheck/AlignScore) have no prompt and are excluded; RAGAS is a
different (claim-decomposition) mechanism and reported separately if needed.

Output: results/citation_relocation_pilot/verdicts_attribution_prompt.jsonl
Resumable on (seed_id, judge).
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import anthropic
from mistralai.client.sdk import Mistral

from jfre.judges._llm_judge_prompt import format_passages
from jfre.judges._retry import retry_on_rate_limit
from jfre.seeds.expertqa import load as load_expertqa

OUT = Path("results/citation_relocation_pilot/verdicts_attribution_prompt.jsonl")

# Attribution-aware faithfulness prompt. Identical content-support definition as
# the deployed judge, PLUS the explicit per-citation attribution rule (case 3 of
# the human rubric). This is the only change from the deployed prompt.
_ATTR_PROMPT = """You are a faithfulness judge for a retrieval-augmented question answering system.

You are given a question, a set of numbered retrieved passages, and a candidate answer that contains [N] citation markers (N is the passage number being cited).

Decide whether the candidate answer is FAITHFUL or UNFAITHFUL.

Definitions:
- FAITHFUL: every factual claim is supported by the passages, AND every citation marker [N] points to a passage that actually supports the specific claim it is attached to.
- UNFAITHFUL: at least one claim is unsupported, contradicts a passage, OR carries a citation [N] where passage N does NOT support that claim — even if some other passage would support it (this is citation misattribution).

Check each [N] marker: read passage N and confirm it supports the claim the marker is attached to. Judge relative to the passages only, not world knowledge.

Question:
{question}

Retrieved passages:
{passages}

Candidate answer:
{answer}

Respond with valid JSON only, no markdown fences:
{{"verdict": "faithful" OR "unfaithful", "reasoning": "one sentence"}}"""


@lru_cache(maxsize=1)
def _anthropic():
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@lru_cache(maxsize=1)
def _mistral():
    return Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def _parse(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        v = str(json.loads(raw).get("verdict", "")).strip().lower()
        return v if v in ("faithful", "unfaithful") else "parse_error"
    except Exception:
        m = re.search(r"(faithful|unfaithful)", raw.lower())
        return m.group(1) if m else "parse_error"


@retry_on_rate_limit()
def _claude(prompt):
    msg = _anthropic().messages.create(
        model="claude-sonnet-4-6", max_tokens=512, temperature=0,
        system=[{"type": "text", "text": "You are a careful faithfulness judge. Respond with valid JSON only.",
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


@retry_on_rate_limit()
def _mistral_call(prompt):
    r = _mistral().chat.complete(
        model="mistral-large-latest", temperature=0, max_tokens=512,
        messages=[{"role": "system", "content": "You are a careful faithfulness judge. Respond with valid JSON only."},
                  {"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content or ""


# FaithJudge-style: same attribution prompt but keep its strict-judge framing.
@retry_on_rate_limit()
def _faithjudge_call(prompt):
    msg = _anthropic().messages.create(
        model="claude-sonnet-4-6", max_tokens=512, temperature=0,
        system=[{"type": "text", "text": "You are a strict faithfulness judge attentive to citation attribution. Respond with valid JSON only.",
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


JUDGES = {
    "claude_sonnet_4_6": _claude,
    "mistral_large_2": _mistral_call,
    "faithjudge_style_sonnet": _faithjudge_call,
}


def main():
    # 100 accepted citation seeds: passages from expertqa loader; perturbed answer
    # (with relocated [N]) from the citation perturbations file.
    seedmap = {s.seed_id: s for s in load_expertqa(n=1500)}
    perts = {}
    for line in Path("results/citation_relocation_pilot/perturbations.jsonl").open():
        r = json.loads(line)
        if r.get("rule_passed") and r["operator"] == "citation_relocation":
            perts[r["seed_id"]] = r
    accepted = [json.loads(l)["seed_id"] for l in
                Path("results/citation_relocation_pilot/seeds.jsonl").open()
                if json.loads(l).get("accepted")]
    cells = [sid for sid in accepted if sid in perts and sid in seedmap]

    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["seed_id"], r["judge"]))

    work = [(sid, j) for sid in cells for j in JUDGES if (sid, j) not in done]
    print(f"{len(work)} (cell, judge) calls to make ({len(cells)} cells x {len(JUDGES)} judges)", flush=True)

    with OUT.open("a") as f:
        for i, (sid, judge) in enumerate(work, 1):
            seed = seedmap[sid]
            answer = perts[sid]["perturbed_answer"]
            prompt = _ATTR_PROMPT.format(
                question=seed.question,
                passages=format_passages(seed.passages),
                answer=answer,
            )
            try:
                raw = JUDGES[judge](prompt)
                verdict = _parse(raw)
            except Exception as e:
                verdict = "error"
                raw = str(e)[:160]
            f.write(json.dumps({"seed_id": sid, "operator": "citation_relocation",
                                "judge": judge, "verdict": verdict,
                                "prompt_type": "attribution_aware"}) + "\n")
            f.flush()
            if i % 30 == 0:
                print(f"  {i}/{len(work)}", flush=True)

    print("Done. Compare with: scripts/compare_attribution_prompt.py", flush=True)


if __name__ == "__main__":
    main()
