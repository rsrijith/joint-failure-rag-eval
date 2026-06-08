"""Capped Claude (+FaithJudge-style, +Mistral if funded) confirmation arm for the
PubMedQA citation_relocation replication. Primary-judge parity with the ExpertQA result.

Budget-safe: tracks token usage live and HARD-STOPS at COST_CAP_USD (default 2.80,
margin under the user's ~$3.20 Claude balance). 4 cells/seed/judge
(content/attribution x clean/scrambled). Claude+FaithJudge are Anthropic; Mistral is
attempted and skipped gracefully if unfunded.

Run:  python scripts/pubmedqa_claude.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import anthropic

from jfre.judges._llm_judge_prompt import PROMPT_TEMPLATE, format_passages
from jfre.operators.citation_relocation import generate as relocate
from jfre.types import Passage, Seed

CACHE = Path("data/cache/pubmedqa_cited_llama.jsonl")
ACC = Path("results/pubmedqa_accepted.jsonl")
OUT = Path("results/pubmedqa_claude_verdicts.jsonl")
SUMM = Path("results/pubmedqa_claude.txt")
N_CAP = 60
COST_CAP_USD = 2.80
CLAUDE_MODEL = "claude-sonnet-4-6"
IN_PER_M, OUT_PER_M = 3.0, 15.0  # Sonnet pricing

ATTR_PROMPT = """You are a faithfulness judge for a retrieval-augmented question answering system.
You are given a question, a set of numbered retrieved passages, and a candidate answer that contains [N] citation markers (N is the passage number being cited).
Decide whether the candidate answer is FAITHFUL or UNFAITHFUL.
Definitions:
- FAITHFUL: every factual claim is supported by the passages, AND every citation marker [N] points to a passage that actually supports the specific claim it is attached to.
- UNFAITHFUL: at least one claim is unsupported, contradicts a passage, OR carries a citation [N] where passage N does NOT support that claim, even if some other passage would support it (this is citation misattribution).
Check each [N] marker: read passage N and confirm it supports the claim the marker is attached to. Judge relative to the passages only, not world knowledge.
Question:
{question}
Retrieved passages:
{passages}
Candidate answer:
{answer}
Respond with valid JSON only: {{"verdict": "faithful" OR "unfaithful", "reasoning": "one sentence"}}"""

FAITHJUDGE_SYS = "You are a strict faithfulness judge attentive to citation attribution. Respond with valid JSON only."
_cost = {"in": 0, "out": 0}
_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def est_cost():
    return _cost["in"] / 1e6 * IN_PER_M + _cost["out"] / 1e6 * OUT_PER_M


def claude(prompt, system=None):
    kw = {"model": CLAUDE_MODEL, "max_tokens": 256, "temperature": 0,
          "messages": [{"role": "user", "content": prompt}]}
    if system:
        kw["system"] = system
    m = _client.messages.create(**kw)
    _cost["in"] += m.usage.input_tokens
    _cost["out"] += m.usage.output_tokens
    txt = "".join(b.text for b in m.content if b.type == "text").strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt); txt = re.sub(r"\s*```$", "", txt).strip()
    try:
        v = str(json.loads(txt).get("verdict", "")).strip().lower()
    except Exception:
        mm = re.search(r"\b(faithful|unfaithful)\b", txt.lower()); v = mm.group(1) if mm else "parse_error"
    return v if v in ("faithful", "unfaithful") else "parse_error"


def passages_by_seed():
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    out = {}
    for r in ds:
        c = r["context"]["contexts"]
        if len(c) >= 3:
            out[f"pubmedqa-{r['pubid']}"] = ([Passage(text=x, is_relevant=True) for x in c], r["question"])
    return out


def main():
    accepted = [l.strip() for l in ACC.read_text().splitlines() if l.strip()][:N_CAP]
    cache = {json.loads(l)["seed_id"]: json.loads(l) for l in CACHE.open()}
    pmap = passages_by_seed()
    vf = OUT.open("w")
    stopped = False
    done = 0
    for sid in accepted:
        if est_cost() > COST_CAP_USD:
            print(f"COST CAP hit at ${est_cost():.2f} after {done} seeds; stopping.", flush=True)
            stopped = True
            break
        passages, question = pmap[sid]
        cited = cache[sid]["cited_answer"]
        seed = Seed(seed_id=sid, source="expertqa", question=question, passages=passages,
                    gold_answer=cited, metadata={"cited_answer": cited})
        scr = relocate(seed)
        if not scr.rule_passed:
            continue
        p = format_passages(passages)
        content_clean = PROMPT_TEMPLATE.format(question=question, passages=p, answer=cited)
        content_scr = PROMPT_TEMPLATE.format(question=question, passages=p, answer=scr.perturbed_answer)
        attr_clean = ATTR_PROMPT.format(question=question, passages=p, answer=cited)
        attr_scr = ATTR_PROMPT.format(question=question, passages=p, answer=scr.perturbed_answer)
        cells = {
            ("claude_sonnet_4_6", "content", "clean"): (content_clean, None),
            ("claude_sonnet_4_6", "content", "scrambled"): (content_scr, None),
            ("claude_sonnet_4_6", "attribution", "clean"): (attr_clean, None),
            ("claude_sonnet_4_6", "attribution", "scrambled"): (attr_scr, None),
            ("faithjudge_style_sonnet", "content", "clean"): (content_clean, FAITHJUDGE_SYS),
            ("faithjudge_style_sonnet", "content", "scrambled"): (content_scr, FAITHJUDGE_SYS),
            ("faithjudge_style_sonnet", "attribution", "clean"): (attr_clean, FAITHJUDGE_SYS),
            ("faithjudge_style_sonnet", "attribution", "scrambled"): (attr_scr, FAITHJUDGE_SYS),
        }
        for (judge, prompt_type, tag), (prompt, sys) in cells.items():
            v = claude(prompt, sys)
            vf.write(json.dumps({"seed_id": sid, "judge": judge, "prompt_type": prompt_type,
                                 "tag": tag, "verdict": v}) + "\n")
        vf.flush()
        done += 1
        if done % 5 == 0:
            print(f"  {done} seeds, est ${est_cost():.2f}", flush=True)
    vf.close()

    rows = [json.loads(l) for l in OUT.open()]
    def fr(judge, pt, tag):
        xs = [r["verdict"] for r in rows if r["judge"] == judge and r["prompt_type"] == pt
              and r["tag"] == tag and r["verdict"] in ("faithful", "unfaithful")]
        return (round(100 * sum(v == "faithful" for v in xs) / len(xs), 1), len(xs)) if xs else (None, 0)

    lines = ["PubMedQA Claude/FaithJudge confirmation arm — citation_relocation 2x2",
             "=" * 70, f"seeds scored: {done}   est cost: ${est_cost():.2f}"
             + ("  [STOPPED AT CAP]" if stopped else ""), ""]
    for j in ("claude_sonnet_4_6", "faithjudge_style_sonnet"):
        cc = fr(j, "content", "clean"); cs = fr(j, "content", "scrambled")
        ac = fr(j, "attribution", "clean"); as_ = fr(j, "attribution", "scrambled")
        gain = (cs[0] - as_[0]) if (cs[0] is not None and as_[0] is not None) else None
        lines += [f"{j}  (n={cs[1]})",
                  f"  content:clean {cc[0]}  content:scrambled {cs[0]}  "
                  f"attribution:clean {ac[0]}  attribution:scrambled {as_[0]}",
                  f"  attribution gain (content-scram - attr-scram FNR) = {gain:+}pp" if gain is not None else "  (incomplete)",
                  ""]
    SUMM.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
