"""F7 per-(claim, cited-passage) NLI ablation on the PubMedQA accepted seeds (FREE).

Replicates §5.4(c) / Table 5 on biomedical text: does per-citation NLI scoring recover
misattribution sensitivity (pooled FNR -> per-citation FNR), and what is the operator's
contamination rate when the relocated passage is another section of the SAME abstract
(expected higher than ExpertQA because sections are topically tight)?

Run:  JFRE_DEVICE=mps python scripts/pubmedqa_f7.py
"""

from __future__ import annotations

# AlignScore's package does `from transformers import AdamW` (removed in transformers 4.x).
# Patch it before any alignscore import, and import transformers first (working order).
import transformers as _tf
import torch as _torch
if not hasattr(_tf, "AdamW"):
    _tf.AdamW = _torch.optim.AdamW

import json
import random
import re
from pathlib import Path

from jfre.types import Passage, Seed

_CIT = re.compile(r"\[(\d+)\]")
CACHE = Path("data/cache/pubmedqa_cited_llama.jsonl")
ACC = Path("results/pubmedqa_accepted.jsonl")
OUT = Path("results/pubmedqa_f7.txt")
THRESH = 0.5


def _derangement(indices, rng):
    if len(indices) < 2:
        return indices[:]
    for _ in range(100):
        sh = indices[:]
        rng.shuffle(sh)
        if all(a != b for a, b in zip(indices, sh)):
            return sh
    return indices[1:] + indices[:1]


def _sigma(seed_id, cited):
    distinct = sorted({int(m) for m in _CIT.findall(cited)})
    return dict(zip(distinct, _derangement(distinct, random.Random(f"42-{seed_id}"))))


def _claims(cited):
    units, last = [], 0
    for m in _CIT.finditer(cited):
        text = _CIT.sub("", cited[last:m.start()]).strip()
        if text:
            units.append((text, int(m.group(1))))
        last = m.end()
    return units


def passages_by_seed():
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    out = {}
    for r in ds:
        ctxs = r["context"]["contexts"]
        if len(ctxs) < 3:
            continue
        out[f"pubmedqa-{r['pubid']}"] = [Passage(text=c, is_relevant=True) for c in ctxs]
    return out


def main():
    accepted = [l.strip() for l in ACC.read_text().splitlines() if l.strip()]
    cache = {json.loads(l)["seed_id"]: json.loads(l) for l in CACHE.open()}
    pmap = passages_by_seed()
    print(f"accepted={len(accepted)}", flush=True)

    work = {}
    for sid in accepted:
        cited = cache[sid]["cited_answer"]
        passages = pmap.get(sid)
        if not passages:
            continue
        sigma = _sigma(sid, cited)
        rows = []
        for claim, n in _claims(cited):
            if 1 <= n <= len(passages) and 1 <= sigma.get(n, n) <= len(passages):
                rows.append({"claim": claim, "p_clean": passages[n - 1].text,
                             "p_scr": passages[sigma[n] - 1].text})
        if rows:
            work[sid] = rows
    n_unit = sum(len(r) for r in work.values())
    print(f"{n_unit} claim-citation units across {len(work)} seeds", flush=True)

    from jfre.judges.hhem_judge import _model as hhem_model
    from jfre.judges.minicheck_judge import _scorer as mc_scorer, _score_via_minicheck, _score_via_transformers
    from jfre.judges.alignscore_judge import _scorer as al_scorer

    hh = hhem_model()
    def hhem(prem, hyp):
        o = hh.predict([(prem, hyp)]); return float(o[0] if hasattr(o, "__getitem__") else o)
    mcb, mco = mc_scorer()
    def minicheck(prem, hyp):
        return _score_via_minicheck(mco, prem, hyp) if mcb == "minicheck" else _score_via_transformers(mco, prem, hyp)
    al = al_scorer()
    def align(prem, hyp):
        return float(al.score(contexts=[prem], claims=[hyp])[0])

    JUDGES = {"hhem_2_1_open": hhem, "minicheck_flan_t5_large": minicheck, "alignscore_large": align}
    POOLED = {}  # filled from pubmedqa_free if present
    fr = Path("results/pubmedqa_free_verdicts.jsonl")
    if fr.exists():
        rows = [json.loads(l) for l in fr.open()]
        acc = set(accepted)
        for j in JUDGES:
            xs = [r["verdict"] for r in rows if r["judge"] == j and r["tag"] == "scrambled"
                  and r["seed_id"] in acc and r["verdict"] in ("faithful", "unfaithful")]
            POOLED[j] = round(100 * sum(v == "faithful" for v in xs) / len(xs), 1) if xs else None

    lines = ["PubMedQA F7: per-(claim, cited-passage) NLI ablation", "=" * 66,
             f"accepted seeds={len(work)}, claim-citation units={n_unit}", ""]
    for jn, score in JUDGES.items():
        seed_scr_faithful = seed_clean_faithful = 0
        claim_scr_sup = claim_clean_sup = 0
        n_seed = n_u = 0
        for sid, rows in work.items():
            n_seed += 1; c_all = s_all = True
            for r in rows:
                n_u += 1
                c_ok = score(r["p_clean"], r["claim"]) >= THRESH
                s_ok = score(r["p_scr"], r["claim"]) >= THRESH
                claim_clean_sup += c_ok; claim_scr_sup += s_ok
                c_all &= c_ok; s_all &= s_ok
            seed_clean_faithful += c_all; seed_scr_faithful += s_all
        per_fnr = 100 * seed_scr_faithful / n_seed
        clean_pass = 100 * seed_clean_faithful / n_seed
        contam = 100 * claim_scr_sup / n_u
        pooled = POOLED.get(jn)
        lines.append(f"{jn}")
        lines.append(f"  pooled FNR (deployed)  : {pooled}%")
        lines.append(f"  per-citation FNR       : {per_fnr:.1f}%   (Δ {(pooled - per_fnr):+.1f}pp)" if pooled is not None
                     else f"  per-citation FNR       : {per_fnr:.1f}%")
        lines.append(f"  clean per-citation pass: {clean_pass:.1f}%")
        lines.append(f"  contamination rate     : {contam:.1f}%")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
