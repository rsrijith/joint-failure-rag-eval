"""Cross-model citation annotation with Llama-3.3-70B via Groq (circularity control).

Independent annotator: Meta Llama-3.3-70B is a different model family from every
judge (Claude/Mistral), so re-annotating with it breaks the Claude-annotates /
Claude-judges circularity in the attribution-prompt experiment. Groq free tier
allows ~1000 requests/day (vs Gemini's 20/day), so all 100 finish in one run.

Reads:  validation/gemini_annotation_input.jsonl  (model-agnostic input)
Writes: data/cache/expertqa_cited_llama.jsonl     (seed_id, cited_answer, distinct_indices)
Resumable; skips only SUCCESSFUL records. Uses GROQ_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

MODEL = "llama-3.3-70b-versatile"
IN = Path("validation/gemini_annotation_input.jsonl")
OUT = Path("data/cache/expertqa_cited_llama.jsonl")
_CIT_RE = re.compile(r"\[(\d+)\]")

PROMPT = """You are annotating an answer with citation markers.

You are given a QUESTION, a list of numbered PASSAGES, and an ANSWER that has no citations. For each factual claim in the answer, insert a citation marker [N] immediately after the claim, where N is the number of the passage that DIRECTLY supports that claim.

Rules:
- Only cite a passage that explicitly contains the cited claim's content.
- One marker per claim, placed at the end of the sentence or clause containing the claim.
- If several passages support a claim, pick the single most directly relevant one.
- If no passage supports a claim, leave it without a citation.
- Do NOT change any of the answer's wording. Only insert [N] markers.

Return ONLY the annotated answer text with [N] markers inserted. No commentary, no JSON, no markdown fences.

QUESTION:
{question}

PASSAGES:
{passages}

ANSWER (no citations):
{answer}"""


def annotate(question, passages, answer, key):
    for attempt in range(10):
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODEL, "temperature": 0, "max_tokens": 2048,
                  "messages": [{"role": "user", "content": PROMPT.format(
                      question=question, passages=passages, answer=answer)}]},
            timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 429:
            time.sleep(min(10 * (attempt + 1), 60))
            continue
        if r.status_code in (500, 502, 503):
            time.sleep(min(2 ** attempt, 30))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    raise RuntimeError("retries exhausted")


def main():
    key = os.environ["GROQ_API_KEY"]
    rows = [json.loads(l) for l in IN.open()]
    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            if r.get("cited_answer") and not r.get("error"):
                done.add(r["seed_id"])
    todo = [r for r in rows if r["seed_id"] not in done]
    print(f"{len(todo)} seeds to annotate with {MODEL}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        for i, r in enumerate(todo, 1):
            try:
                cited = annotate(r["question"], r["passages"], r["gold_answer"], key)
                # Llama sometimes wraps in fences or adds a preamble; strip fences.
                cited = re.sub(r"^```(?:\w+)?\s*", "", cited.strip())
                cited = re.sub(r"\s*```$", "", cited).strip()
                n_p = r["n_passages"]
                idx = sorted({int(m) for m in _CIT_RE.findall(cited) if 1 <= int(m) <= n_p})
                rec = {"seed_id": r["seed_id"], "cited_answer": cited,
                       "distinct_indices": idx, "annotator": MODEL}
            except Exception as e:
                rec = {"seed_id": r["seed_id"], "cited_answer": "",
                       "distinct_indices": [], "error": str(e)[:160], "annotator": MODEL}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    recs = [json.loads(l) for l in OUT.open()]
    ok = [r for r in recs if r.get("cited_answer") and not r.get("error")]
    usable = sum(1 for r in ok if len(r.get("distinct_indices", [])) >= 2)
    print(f"Done. {len(ok)} annotated, {usable} usable (>=2 distinct citations).", flush=True)


if __name__ == "__main__":
    main()
