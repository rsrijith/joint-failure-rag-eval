"""F7: per-(claim, cited-passage) NLI ablation for the 3 NLI judges.

Reviewer-convergent ask: the paper claims NLI judges are STRUCTURALLY blind to
citation misattribution because they pool all passages into one premise. ALCE
(Gao et al. 2023) runs NLI per (claim, cited-passage) pair successfully, so the
"structural" claim is falsifiable unless we test the per-citation configuration.

This script re-runs HHEM / MiniCheck / AlignScore on each cited seed in
PER-CITATION mode: for every claim (the text a [N] marker is attached to), score
the claim against ONLY its cited passage, clean (correct [N]) and scrambled
(citation_relocation derangement). A per-citation judge calls the answer faithful
iff every claim is entailed by its cited passage.

Two outputs:
  1. Per-citation scrambled FNR per judge  vs  the pooled FNR from Table 1
     (HHEM 66, MiniCheck 71, AlignScore 84). If per-citation FNR drops sharply,
     "structural" must be softened to "structural under pooled-premise deployment."
  2. Derangement-contamination rate: fraction of scrambled claims whose relocated
     passage STILL entails the claim (the operator deranges to a different passage
     but does not verify non-support). This bounds how many citation_relocation
     cells are not true attribution probes (R1's methodological flag).

Run:  JFRE_DEVICE=mps python scripts/f7_per_citation_nli.py
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY", "unused-cache-read"))

from jfre.seeds.expertqa_cited import load as load_cited

_CITATION_RE = re.compile(r"\[(\d+)\]")
_OUT = Path("results/f7_per_citation_nli.jsonl")
_SUMMARY = Path("results/f7_per_citation_nli.txt")
POOLED_FNR = {"hhem_2_1_open": 61, "minicheck_flan_t5_large": 70, "alignscore_large": 87}  # n=250 Table 1 NLI deployed


def _derangement(indices, rng):
    """Mirror jfre/operators/citation_relocation.py exactly."""
    if len(indices) < 2:
        return indices[:]
    for _ in range(100):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(indices, shuffled)):
            return shuffled
    return indices[1:] + indices[:1]


def _sigma(seed_id, cited):
    distinct = sorted({int(m) for m in _CITATION_RE.findall(cited)})
    rng = random.Random(f"42-{seed_id}")
    permuted = _derangement(distinct, rng)
    return dict(zip(distinct, permuted))


def _claims(cited):
    """One unit per [N] occurrence: (claim_text, cited_index).

    claim_text = the span the marker is attached to (since the previous marker),
    with all [N] markers stripped. This is literally the text each citation backs.
    """
    units = []
    last = 0
    for m in _CITATION_RE.finditer(cited):
        span = cited[last:m.start()]
        text = _CITATION_RE.sub("", span).strip()
        if text:
            units.append((text, int(m.group(1))))
        last = m.end()
    return units


def _passage_text(passages, idx):
    if 1 <= idx <= len(passages):
        return passages[idx - 1].text
    return None


# ---- raw NLI scorers (premise, hypothesis) -> prob in [0,1] -----------------

def _hhem_scorer():
    from jfre.judges.hhem_judge import _model
    mdl = _model()
    def s(premise, hyp):
        out = mdl.predict([(premise, hyp)])
        return float(out[0] if hasattr(out, "__getitem__") else out)
    return s


def _minicheck_scorer():
    from jfre.judges.minicheck_judge import _scorer, _score_via_minicheck, _score_via_transformers
    backend, obj = _scorer()
    def s(premise, hyp):
        if backend == "minicheck":
            return _score_via_minicheck(obj, premise, hyp)
        return _score_via_transformers(obj, premise, hyp)
    return s


def _alignscore_scorer():
    from jfre.judges.alignscore_judge import _scorer
    sc = _scorer()
    def s(premise, hyp):
        return float(sc.score(contexts=[premise], claims=[hyp])[0])
    return s


JUDGES = {
    "hhem_2_1_open": _hhem_scorer,
    "minicheck_flan_t5_large": _minicheck_scorer,
    "alignscore_large": _alignscore_scorer,
}
THRESH = 0.5


def main():
    accepted = {json.loads(l)["seed_id"]
                for l in open("results/citation_relocation_pilot/seeds.jsonl")
                if json.loads(l).get("accepted")}
    seeds = {s.seed_id: s for s in load_cited(470)
             if s.seed_id in accepted and s.metadata.get("cited_answer")}
    print(f"Loaded {len(seeds)}/{len(accepted)} accepted cited seeds with passages")

    # Pre-build per-seed claim units + clean/scrambled premise indices.
    work = {}
    for sid, seed in seeds.items():
        cited = seed.metadata["cited_answer"]
        sigma = _sigma(sid, cited)
        units = _claims(cited)
        rows = []
        for claim, n in units:
            p_clean = _passage_text(seed.passages, n)
            p_scr = _passage_text(seed.passages, sigma.get(n, n))
            if p_clean is None or p_scr is None:
                continue
            rows.append({"claim": claim, "cited_idx": n, "scr_idx": sigma.get(n, n),
                         "p_clean": p_clean, "p_scr": p_scr})
        if rows:
            work[sid] = rows

    n_claims = sum(len(r) for r in work.values())
    print(f"{n_claims} claim-citation units across {len(work)} seeds")

    out_f = _OUT.open("w")
    summary = []
    for jname, mk in JUDGES.items():
        print(f"\n=== {jname} ===")
        try:
            score = mk()
        except Exception as e:
            print(f"  SKIP {jname}: {e!r}")
            summary.append((jname, None))
            continue

        seed_clean_faithful = 0   # all claims entailed by correct passage
        seed_scr_faithful = 0     # all claims entailed by relocated passage (= FNR numerator)
        claim_clean_sup = 0
        claim_scr_sup = 0         # contamination numerator
        n_seed = 0
        n_unit = 0
        for sid, rows in work.items():
            n_seed += 1
            clean_all = True
            scr_all = True
            for r in rows:
                n_unit += 1
                sc_clean = score(r["p_clean"], r["claim"])
                sc_scr = score(r["p_scr"], r["claim"])
                c_ok = sc_clean >= THRESH
                s_ok = sc_scr >= THRESH
                claim_clean_sup += int(c_ok)
                claim_scr_sup += int(s_ok)
                clean_all = clean_all and c_ok
                scr_all = scr_all and s_ok
                out_f.write(json.dumps({"judge": jname, "seed_id": sid, "cited_idx": r["cited_idx"],
                                        "scr_idx": r["scr_idx"], "score_clean": round(sc_clean, 4),
                                        "score_scr": round(sc_scr, 4)}) + "\n")
            seed_clean_faithful += int(clean_all)
            seed_scr_faithful += int(scr_all)
        out_f.flush()

        per_cit_fnr = 100 * seed_scr_faithful / n_seed
        clean_pass = 100 * seed_clean_faithful / n_seed
        contam = 100 * claim_scr_sup / n_unit
        clean_claim_sup = 100 * claim_clean_sup / n_unit
        pooled = POOLED_FNR[jname]
        summary.append((jname, {
            "pooled_fnr": pooled, "per_citation_fnr": round(per_cit_fnr, 1),
            "delta": round(pooled - per_cit_fnr, 1),
            "clean_pass_per_cit": round(clean_pass, 1),
            "contamination_rate": round(contam, 1),
            "clean_claim_support": round(clean_claim_sup, 1),
            "n_seed": n_seed, "n_unit": n_unit,
        }))
        print(f"  pooled FNR {pooled}%  ->  per-citation FNR {per_cit_fnr:.1f}%  (Δ {pooled-per_cit_fnr:+.1f}pp)")
        print(f"  clean per-citation pass-rate {clean_pass:.1f}%  | contamination {contam:.1f}%  "
              f"| clean claim-support {clean_claim_sup:.1f}%")

    out_f.close()

    with _SUMMARY.open("w") as f:
        f.write("F7: per-(claim, cited-passage) NLI ablation\n")
        f.write("=" * 70 + "\n")
        f.write("Per-citation mode: each claim scored against ONLY its cited passage.\n")
        f.write("Answer = faithful iff every claim entailed by its cited passage.\n")
        f.write("Pooled FNR = the deployed whole-premise number from Table 1.\n")
        f.write("Contamination = % of scrambled claims whose relocated passage STILL\n")
        f.write("  entails the claim (operator deranges but does not verify non-support).\n\n")
        for jname, d in summary:
            if d is None:
                f.write(f"{jname}: SKIPPED\n"); continue
            f.write(f"{jname}  (n_seed={d['n_seed']}, n_unit={d['n_unit']})\n")
            f.write(f"  pooled FNR (deployed)      : {d['pooled_fnr']}%\n")
            f.write(f"  per-citation FNR           : {d['per_citation_fnr']}%   (Δ {d['delta']:+}pp)\n")
            f.write(f"  clean per-citation pass    : {d['clean_pass_per_cit']}%\n")
            f.write(f"  clean claim-support rate   : {d['clean_claim_support']}%\n")
            f.write(f"  contamination rate         : {d['contamination_rate']}%\n\n")
        f.write(json.dumps([s for _, s in summary if s], indent=2))
    print(f"\nWrote {_SUMMARY}")


if __name__ == "__main__":
    main()
