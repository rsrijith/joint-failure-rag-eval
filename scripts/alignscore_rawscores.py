"""Capture AlignScore raw scores incrementally (crash-safe, MPS-friendly).

Writes one JSONL row per scored item to results/alignscore_rawscores.jsonl as
it goes, freeing MPS/torch memory between calls. Resumable: skips items already
in the output. Analyze with analyze_alignscore_rawscores.py.

Run:  JFRE_DEVICE=mps python -u scripts/alignscore_rawscores.py
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

from jfre.judges import alignscore_judge
from jfre.judges._llm_judge_prompt import format_passages
from jfre.seeds.expertqa import load as load_expertqa
from jfre.seeds.hotpotqa import load as load_hotpotqa

RNG = random.Random(7)
N_CLEAN = 80
N_PER_OP = 30
ADVERSARIAL_OPS = ["entity_swap", "numeric_drift", "hedge_insertion",
                   "distractor_parroting", "citation_relocation"]
OUT = Path("results/alignscore_rawscores.jsonl")


def _free_memory():
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def build_seed_map():
    smap = {}
    for s in load_hotpotqa(n=1000):
        smap[s.seed_id] = s
    for s in load_expertqa(n=1500):
        smap[s.seed_id] = s
    return smap


def load_perts():
    out = {}
    for base in ("results/preview_pilot/perturbations.jsonl",
                 "results/citation_relocation_pilot/perturbations.jsonl"):
        p = Path(base)
        if not p.exists():
            continue
        for line in p.open():
            r = json.loads(line)
            if r.get("rule_passed"):
                out[(r["seed_id"], r["operator"])] = r
    return out


def accepted_ids():
    ids = set()
    for base in ("results/preview_pilot/seeds.jsonl",
                 "results/citation_relocation_pilot/seeds.jsonl"):
        for line in Path(base).open():
            s = json.loads(line)
            if s.get("accepted"):
                ids.add(s["seed_id"])
    return ids


def already_done():
    done = set()
    if OUT.exists():
        for line in OUT.open():
            r = json.loads(line)
            done.add((r["seed_id"], r["condition"]))
    return done


def main():
    smap = build_seed_map()
    perts = load_perts()
    accepted = accepted_ids()
    done = already_done()

    # Build work list: (seed_id, condition, operator, answer)
    work = []
    clean_ids = [s for s in accepted if s in smap]
    RNG.shuffle(clean_ids)
    for sid in clean_ids[:N_CLEAN]:
        work.append((sid, "clean", "clean", smap[sid].gold_answer))
    for op in ADVERSARIAL_OPS:
        cells = [(sid, o) for (sid, o) in perts if o == op and sid in smap]
        RNG.shuffle(cells)
        for sid, o in cells[:N_PER_OP]:
            work.append((sid, f"perturbed:{op}", op, perts[(sid, o)]["perturbed_answer"]))

    work = [w for w in work if (w[0], w[1]) not in done]
    print(f"{len(work)} items to score (device={alignscore_judge._select_device()})", flush=True)

    with OUT.open("a") as f:
        for i, (sid, cond, op, answer) in enumerate(work, 1):
            seed = smap[sid]
            premise_words = len(format_passages(seed.passages).split())
            try:
                v = alignscore_judge.score(seed, answer, operator="clean")
                row = {
                    "seed_id": sid, "condition": cond, "operator": op,
                    "raw_score": v.raw_score, "premise_words": premise_words,
                    "error": v.judge_metadata.get("error"),
                }
            except Exception as e:
                row = {"seed_id": sid, "condition": cond, "operator": op,
                       "raw_score": None, "premise_words": premise_words,
                       "error": str(e)[:160]}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(work)} scored", flush=True)
            _free_memory()

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
