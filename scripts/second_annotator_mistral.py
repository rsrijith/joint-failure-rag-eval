"""Mistral as an independent SECOND ANNOTATOR for the human validation.

Re-labels the same 150 validation items the human labeled, but with the HUMAN
RUBRIC prompt (equivalence for Task A, attribution-aware faithfulness for Task B)
-- NOT the faithfulness-judge prompt. This gives a human-vs-model second-annotator
agreement (proxy inter-annotator agreement) so the single-human gold is not the
only signal.

Note: Mistral is also one of the 7 judges, but here it plays the annotator role
with a different prompt. Reported as IAA proxy, not as judge evidence.

Resumable: caches to validation/second_annotator_mistral.jsonl.
Run:  python scripts/second_annotator_mistral.py
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

from mistralai.client.sdk import Mistral

from jfre.judges._retry import retry_on_rate_limit
from jfre.seeds.expertqa import load as load_expertqa
from jfre.seeds.hotpotqa import load as load_hotpotqa

_MODEL = "mistral-large-latest"
OUT = Path("validation/second_annotator_mistral.jsonl")

_EQUIV_PROMPT = """You are checking whether two answers convey the exact same factual content.

Answer B conveys the EXACT SAME factual content as Answer A if the facts are identical, even when wording, clause order, or voice (active/passive) differ. It is NOT the same if Answer B adds, drops, or changes any fact: a named entity, number, date, place, quantity, or a qualifier/scope word (all vs some, always vs often, caused vs associated with).

Answer A:
{a}

Answer B:
{b}

Return JSON only: {{"equivalent": "yes" OR "no"}}"""

_FAITH_PROMPT = """You are judging whether a candidate answer is FAITHFUL or UNFAITHFUL to retrieved passages.

FAITHFUL = every factual claim in the answer is supported by the passages, AND every citation marker [N], if present, points to a passage that actually supports the claim it is attached to.
UNFAITHFUL = at least one of: (1) a claim is not supported by any passage, (2) a claim contradicts a passage, (3) a claim carries a citation [N] but passage N does not support that claim (misattribution) even if another passage would, (4) the answer relies on content from a passage that does not address the question.
Judge relative to the passages only, not world knowledge.

Question:
{question}

Retrieved passages:
{passages}

Candidate answer:
{answer}

Return JSON only: {{"verdict": "faithful" OR "unfaithful"}}"""


@lru_cache(maxsize=1)
def _client():
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    return Mistral(api_key=key)


@retry_on_rate_limit()
def _ask(prompt):
    r = _client().chat.complete(
        model=_MODEL, temperature=0, max_tokens=128,
        messages=[{"role": "system", "content": "You annotate carefully. Respond with valid JSON only."},
                  {"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content or ""


def _parse(raw, field):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return str(json.loads(raw).get(field, "")).strip().lower()
    except Exception:
        m = re.search(r'(faithful|unfaithful|yes|no)', raw.lower())
        return m.group(1) if m else "parse_error"


def build_passage_map():
    pm = {}
    for s in load_hotpotqa(n=1000):
        pm[s.seed_id] = s.passages
    for s in load_expertqa(n=1500):
        pm[s.seed_id] = s.passages
    return pm


def load_perts():
    out = {}
    for base in ("results/preview_pilot/perturbations.jsonl",
                 "results/citation_relocation_pilot/perturbations.jsonl"):
        for line in Path(base).open():
            r = json.loads(line)
            if r.get("rule_passed"):
                out[(r["seed_id"], r["operator"])] = r
    return out


def main():
    key = json.load(open("validation/_answer_key.json"))
    pm = build_passage_map()
    perts = load_perts()

    done = set()
    if OUT.exists():
        for line in OUT.open():
            done.add(json.loads(line)["item_id"])

    with OUT.open("a") as f:
        for iid, meta in key.items():
            if iid in done:
                continue
            sid, op = meta["seed_id"], meta["operator"]
            rec = perts.get((sid, op))
            if rec is None:
                continue
            if iid.startswith("PN-"):  # Task A: equivalence
                raw = _ask(_EQUIV_PROMPT.format(a=rec["gold_answer"], b=rec["perturbed_answer"]))
                label = _parse(raw, "equivalent")
            else:  # Task B: faithfulness with passages
                passages = pm.get(sid, [])
                ptxt = "\n\n".join(f"[PASSAGE {i+1}] {p.text}" for i, p in enumerate(passages))
                raw = _ask(_FAITH_PROMPT.format(question=rec["question"], passages=ptxt,
                                                answer=rec["perturbed_answer"]))
                label = _parse(raw, "verdict")
            f.write(json.dumps({"item_id": iid, "seed_id": sid, "operator": op, "label": label}) + "\n")
            f.flush()
            print(f"  {iid} [{op}] -> {label}", flush=True)

    print(f"\nDone. {OUT}")


if __name__ == "__main__":
    main()
