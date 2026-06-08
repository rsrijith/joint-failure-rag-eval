"""Attribution-prompt experiment on GEMINI-annotated citations (circularity control).

Re-runs the content-vs-attribution 2x2 on citations produced by Gemini (a non-judge
model) instead of Claude. If the attribution-prompt Δ holds here, the
Claude-annotates / Claude-judges circularity objection is answered.

For each usable Gemini-cited seed (>=2 distinct [N] indices):
  - clean    = the Gemini cited answer (correct attributions)
  - scrambled = citation_relocation derangement of it (mis-attributions)
score each under: content prompt (render_prompt) and attribution-aware prompt,
for the 3 LLM judges (Claude, Mistral, FaithJudge-style). 4 conditions x 3 judges.

Output: results/citation_relocation_pilot/verdicts_gemini_attribution.jsonl
Resumable on (seed_id, judge, condition, prompt_type).
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import anthropic
import requests
from mistralai.client.sdk import Mistral

from jfre.judges._llm_judge_prompt import format_passages, render_prompt
from jfre.operators import citation_relocation
from jfre.seeds.expertqa import load as load_expertqa

# Independent-annotator cited answers. Default to the Llama-3.3-70B set (Groq,
# 1000/day) since Gemini's free tier caps at 20/day. Override with JFRE_CITED.
OUT = Path("results/citation_relocation_pilot/verdicts_crossmodel_attribution.jsonl")
GEMINI_CITED = Path(os.environ.get("JFRE_CITED", "data/cache/expertqa_cited_llama.jsonl"))

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


def _retry(fn):
    for a in range(6):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            if any(t in msg for t in ("rate", "429", "503", "500", "overloaded", "timeout")):
                time.sleep(min(2 ** a, 30)); continue
            raise
    raise RuntimeError("retries exhausted")


def _claude(prompt):
    def go():
        m = _anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=512, temperature=0,
            system=[{"type": "text", "text": "You are a careful faithfulness judge. Respond with valid JSON only.",
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in m.content if b.type == "text")
    return _retry(go)


def _faithjudge(prompt):
    def go():
        m = _anthropic().messages.create(
            model="claude-sonnet-4-6", max_tokens=512, temperature=0,
            system=[{"type": "text", "text": "You are a strict faithfulness judge attentive to citation attribution. Respond with valid JSON only.",
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in m.content if b.type == "text")
    return _retry(go)


def _mistral_call(prompt):
    def go():
        r = _mistral().chat.complete(
            model="mistral-large-latest", temperature=0, max_tokens=512,
            messages=[{"role": "system", "content": "You are a careful faithfulness judge. Respond with valid JSON only."},
                      {"role": "user", "content": prompt}])
        return r.choices[0].message.content or ""
    return _retry(go)


JUDGES = {"claude_sonnet_4_6": _claude, "mistral_large_2": _mistral_call,
          "faithjudge_style_sonnet": _faithjudge}


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


def main():
    seedmap = {s.seed_id: s for s in load_expertqa(n=1500)}
    # Gemini cited answers with >=2 distinct indices
    gem = {}
    for line in GEMINI_CITED.open():
        r = json.loads(line)
        if len(r.get("distinct_indices", [])) >= 2 and r.get("cited_answer"):
            gem[r["seed_id"]] = r["cited_answer"]

    # Build clean + scrambled answers per usable seed
    items = []  # (seed_id, seed, clean_answer, scrambled_answer)
    for sid, cited in gem.items():
        seed = seedmap.get(sid)
        if seed is None:
            continue
        seed.metadata["cited_answer"] = cited
        pert = citation_relocation.generate(seed)
        if not pert.rule_passed:
            continue
        items.append((sid, seed, cited, pert.perturbed_answer))

    print(f"{len(items)} usable Gemini-cited seeds (>=2 citations, derangement ok)", flush=True)

    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["seed_id"], r["judge"], r["condition"], r["prompt_type"]))

    conditions = [("clean", lambda c, s: c), ("scrambled", lambda c, s: s)]
    prompts = [("content", render_prompt), ("attribution", None)]  # None => _ATTR_PROMPT

    work = []
    for sid, seed, clean, scr in items:
        for cond_name, pick in conditions:
            answer = pick(clean, scr)
            for j in JUDGES:
                for ptype, _ in prompts:
                    if (sid, j, cond_name, ptype) not in done:
                        work.append((sid, seed, answer, j, cond_name, ptype))
    print(f"{len(work)} calls ({len(items)} seeds x 2 conditions x 2 prompts x 3 judges)", flush=True)

    with OUT.open("a") as f:
        for i, (sid, seed, answer, judge, cond, ptype) in enumerate(work, 1):
            if ptype == "content":
                prompt = render_prompt(seed.question, seed.passages, answer)
            else:
                prompt = _ATTR_PROMPT.format(question=seed.question,
                                             passages=format_passages(seed.passages), answer=answer)
            try:
                verdict = _parse(JUDGES[judge](prompt))
            except Exception as e:
                verdict = "error"
            f.write(json.dumps({"seed_id": sid, "judge": judge, "condition": cond,
                                "prompt_type": ptype, "verdict": verdict,
                                "annotator": "gemini-2.5-flash"}) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"  {i}/{len(work)}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
