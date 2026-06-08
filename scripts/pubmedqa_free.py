"""Biomedical second-domain replication of citation_relocation (PubMedQA) — FREE core.

Groq Llama-3.3-70B is used ONLY for annotation (inserting [N] markers), the one step
that needs an LLM. Pre-filtering and pooled scoring use the 3 LOCAL NLI judges (HHEM,
MiniCheck, AlignScore) on MPS — zero API, fully reliable. The LLM content-vs-attribution
2x2 comes from the separate funded Claude arm (pubmedqa_claude.py); per-citation NLI from
pubmedqa_f7.py. This keeps the replication robust to Groq free-tier rate limits.

PubMedQA passages are sections of ONE abstract: topically tight, a stringent attribution
test. Contamination reported by F7.

Outputs:
  data/cache/pubmedqa_cited_llama.jsonl  (annotation cache; resumable)
  results/pubmedqa_free_verdicts.jsonl   (NLI verdicts, clean + scrambled)
  results/pubmedqa_accepted.jsonl        (accepted seed_ids for F7 + Claude arm)
  results/pubmedqa_free.txt              (NLI pooled-FNR summary)

Run:  JFRE_DEVICE=mps python scripts/pubmedqa_free.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

from jfre.judges._llm_judge_prompt import format_passages
from jfre.operators.citation_relocation import generate as relocate
from jfre.types import Passage, Seed

GROQ_KEY = os.environ["GROQ_API_KEY"]
GROQ_MODEL = "llama-3.3-70b-versatile"
_CIT = re.compile(r"\[(\d+)\]")
CACHE = Path("data/cache/pubmedqa_cited_llama.jsonl")
VERD = Path("results/pubmedqa_free_verdicts.jsonl")
ACC = Path("results/pubmedqa_accepted.jsonl")
SUMM = Path("results/pubmedqa_free.txt")
N_RAW = 220
ACCEPT_TARGET = 40

ANNOTATE = """You are annotating an answer with citation markers.
You are given a QUESTION, a list of numbered PASSAGES, and an ANSWER with no citations. For each factual claim in the answer, insert a citation marker [N] immediately after the claim, where N is the number of the passage that DIRECTLY supports that claim.
Rules:
- Only cite a passage that explicitly contains the cited claim's content.
- One marker per claim, at the end of the sentence/clause containing the claim.
- If several passages support a claim, pick the single most directly relevant one.
- If no passage supports a claim, leave it without a citation.
- Do NOT change the answer's wording. Only insert [N] markers.
Return ONLY the annotated answer text with [N] markers. No commentary, no JSON, no fences.
QUESTION:
{q}
PASSAGES:
{p}
ANSWER (no citations):
{a}"""


def groq(prompt, max_tokens=2048):
    """Robust Groq call. Returns text on success, None on failure.
    On a daily-cap 429 (Retry-After > 300s) returns None IMMEDIATELY (never sleeps
    for hours). Short 429 backoff capped at 70s, max 6 attempts."""
    for attempt in range(6):
        try:
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_KEY}"},
                json={"model": GROQ_MODEL, "temperature": 0, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]}, timeout=90)
        except requests.RequestException:
            time.sleep(4)
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 429:
            ra = r.headers.get("retry-after")
            ra_s = float(ra) if ra and ra.replace(".", "", 1).isdigit() else 30.0
            if ra_s > 300:          # daily token cap: do not wait hours, just skip
                return None
            time.sleep(min(ra_s, 70) + 1)
            continue
        if 400 <= r.status_code < 500:
            return None
        time.sleep(4)
    return None


def load_seeds():
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    seeds = []
    for r in ds:
        ctxs = r["context"]["contexts"]
        if len(ctxs) < 3:
            continue
        seeds.append(Seed(seed_id=f"pubmedqa-{r['pubid']}", source="expertqa",
                          question=r["question"],
                          passages=[Passage(text=c, is_relevant=True) for c in ctxs],
                          gold_answer=r["long_answer"], metadata={}))
        if len(seeds) >= N_RAW:
            break
    return seeds


def load_cache():
    out = {}
    if CACHE.exists():
        for line in CACHE.open():
            rec = json.loads(line)
            out[rec["seed_id"]] = rec
    return out


def main():
    seeds = load_seeds()
    print(f"PubMedQA seeds (>=3 sections): {len(seeds)}", flush=True)

    cache = load_cache()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = consec = 0
    with CACHE.open("a") as cf:
        for i, s in enumerate(seeds):
            if s.seed_id in cache:
                continue
            cited = groq(ANNOTATE.format(q=s.question, p=format_passages(s.passages), a=s.gold_answer))
            if cited is None:
                fail += 1; consec += 1
                if consec >= 3:
                    print(f"Groq appears rate-capped (3 consecutive fails); "
                          f"proceeding with {len(cache)} cached annotations.", flush=True)
                    break
                continue
            consec = 0
            distinct = sorted({int(m) for m in _CIT.findall(cited)})
            rec = {"seed_id": s.seed_id, "cited_answer": cited, "distinct_indices": distinct}
            cf.write(json.dumps(rec) + "\n"); cf.flush(); cache[s.seed_id] = rec
            ok += 1
            time.sleep(1.2)
            if (ok + fail) % 20 == 0:
                print(f"  annotated ok={ok} fail={fail} (of {i+1} seen)", flush=True)
    annotated = {s.seed_id: s for s in seeds
                 if s.seed_id in cache and len(cache[s.seed_id]["distinct_indices"]) >= 2}
    print(f"annotated with >=2 distinct citations: {len(annotated)}", flush=True)

    from jfre.judges import hhem_judge, minicheck_judge, alignscore_judge
    NLI = {"hhem_2_1_open": hhem_judge.score,
           "minicheck_flan_t5_large": minicheck_judge.score,
           "alignscore_large": alignscore_judge.score}

    vf = VERD.open("w")

    def nli_all(seed, answer, tag):
        out = {}
        for jn, fn in NLI.items():
            v = fn(seed, answer, "citation_relocation").verdict
            out[jn] = v
            vf.write(json.dumps({"seed_id": seed.seed_id, "judge": jn, "tag": tag, "verdict": v}) + "\n")
        vf.flush()
        return out

    accepted = []
    for sid, seed in annotated.items():
        if len(accepted) >= ACCEPT_TARGET:
            break
        cited = cache[sid]["cited_answer"]
        seed.metadata["cited_answer"] = cited
        nli_all(seed, cited, "clean")           # record clean (no hard gate)
        pert = relocate(seed)
        if not pert.rule_passed:
            continue
        nli_all(seed, pert.perturbed_answer, "scrambled")
        accepted.append(sid)
        if len(accepted) % 10 == 0:
            print(f"  accepted {len(accepted)}", flush=True)
    vf.close()
    ACC.write_text("\n".join(accepted) + "\n")
    print(f"ACCEPTED: {len(accepted)}", flush=True)

    rows = [json.loads(l) for l in VERD.open()]
    acc = set(accepted)
    by = {}  # (seed,judge) -> {tag:verdict}
    for r in rows:
        if r["seed_id"] in acc:
            by.setdefault((r["seed_id"], r["judge"]), {})[r["tag"]] = r["verdict"]

    def stats(judge):
        clean_f = scr_f = n = cond_n = cond_scr_f = 0
        for (sid, j), d in by.items():
            if j != judge or "clean" not in d or "scrambled" not in d:
                continue
            n += 1
            cf = d["clean"] == "faithful"; sf = d["scrambled"] == "faithful"
            clean_f += cf; scr_f += sf
            if cf:                      # condition on the judge passing the CLEAN answer
                cond_n += 1; cond_scr_f += sf
        return n, clean_f, scr_f, cond_n, cond_scr_f

    lines = ["PubMedQA biomedical replication (LOCAL NLI judges) — citation_relocation",
             "=" * 72, f"accepted seeds: {len(accepted)}", "",
             "Per judge: clean-faithful%, scrambled-faithful% (=pooled FNR),",
             "and CONDITIONAL miss = scrambled-faithful among seeds the judge passed clean:"]
    for j in ("hhem_2_1_open", "minicheck_flan_t5_large", "alignscore_large"):
        n, cf, sf, cn, csf = stats(j)
        cleanp = 100 * cf / n if n else float("nan")
        fnr = 100 * sf / n if n else float("nan")
        cond = 100 * csf / cn if cn else float("nan")
        lines.append(f"  {j:<26} clean-faithful {cleanp:5.1f}%  scrambled-FNR {fnr:5.1f}%  "
                     f"conditional-miss {cond:5.1f}% (n_cond={cn})  [n={n}]")
    SUMM.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nWrote {SUMM}. Next: pubmedqa_f7.py then pubmedqa_claude.py.", flush=True)


if __name__ == "__main__":
    main()
