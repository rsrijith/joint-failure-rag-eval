"""Cross-model citation annotation with Gemini (breaks annotator-judge circularity).

Re-annotates the 100 accepted citation seeds' gold answers with [N] markers using
Gemini 2.5 Flash (a NON-judge model), so the attribution-prompt experiment can be
re-run on citations no judge model produced.

Reads:  validation/gemini_annotation_input.jsonl  (from build_gemini_annotation_input.py)
Writes: data/cache/expertqa_cited_gemini.jsonl     (seed_id, cited_answer, distinct_indices)
Resumable. Uses GOOGLE_API_KEY from the environment. No new dependency (REST via requests).

Run:  export $(grep -v '^#' .env | xargs); python scripts/gemini_annotate.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

MODEL = "gemini-2.5-flash-lite"  # 2.5-flash free tier is only 20 req/day; flash-lite has a higher cap
IN = Path("validation/gemini_annotation_input.jsonl")
OUT = Path("data/cache/expertqa_cited_gemini.jsonl")
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(
            question=question, passages=passages, answer=answer)}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 2048},
    }
    # Free-tier Gemini throttles (per-minute and per-day). On 429 we wait out the
    # per-minute window patiently (up to ~65s) for many attempts; a persistent
    # 429 (daily cap) eventually raises a TransientError so the caller skips
    # caching and a later resume retries it.
    for attempt in range(12):
        r = requests.post(url, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        if r.status_code == 429:
            time.sleep(min(15 * (attempt + 1), 65))  # patient: ride out per-minute reset
            continue
        if r.status_code in (500, 503):
            time.sleep(min(2 ** attempt, 30))
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    raise TransientError("429 throttle persisted (likely daily cap) — resume later")


class TransientError(Exception):
    pass


def main():
    key = os.environ["GOOGLE_API_KEY"]
    rows = [json.loads(l) for l in IN.open()]
    # Resume only skips SUCCESSFUL annotations; error/empty records are retried
    # (so a throttle-poisoned cache entry does not get skipped forever).
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
                n_p = r["n_passages"]
                idx = sorted({int(m) for m in _CIT_RE.findall(cited) if 1 <= int(m) <= n_p})
                rec = {"seed_id": r["seed_id"], "cited_answer": cited,
                       "distinct_indices": idx, "annotator": MODEL}
            except TransientError as e:
                # Daily cap hit: stop cleanly, do NOT cache. Resume after reset.
                print(f"  THROTTLED at {i}/{len(todo)}: {e}. Resume later — progress saved.", flush=True)
                break
            except Exception as e:
                rec = {"seed_id": r["seed_id"], "cited_answer": "",
                       "distinct_indices": [], "error": str(e)[:160], "annotator": MODEL}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    # quick summary
    recs = [json.loads(l) for l in OUT.open()]
    usable = sum(1 for r in recs if len(r.get("distinct_indices", [])) >= 2)
    print(f"Done. {len(recs)} annotated, {usable} usable (>=2 distinct citations).", flush=True)


if __name__ == "__main__":
    main()
