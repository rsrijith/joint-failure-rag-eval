"""Re-score verdicts corrupted by the empty-premise rescore bug.

The clean-pass rescore (scripts/rescore_errored.py) reconstructed seeds from
seeds.jsonl, which does NOT store passages -> every rescored call ran with an
empty premise. This corrupted ALL AlignScore verdicts (~2075 cells) and ~51
non-AlignScore cells that had errored during the pilot.

This script reconstructs seeds WITH passages from the source loaders and
re-scores the affected cells correctly. Writes fresh verdicts to:
    results/<pilot>/verdicts_refixed.jsonl   (one row per re-scored cell)

Resumable: skips (seed, op, judge) already present in the refixed file.
AlignScore runs on MPS if JFRE_DEVICE=mps.

After this completes, scripts/rebuild_verdicts.py merges the refixed verdicts
into a corrected verdicts file.
"""

from __future__ import annotations

import json
import gc
import glob
from pathlib import Path

from jfre.judges import (
    alignscore_judge, claude_judge, faithjudge, hhem_judge,
    minicheck_judge, mistral_judge, ragas_judge,
)
from jfre.types import Passage, Seed
from jfre.seeds.expertqa import load as load_expertqa
from jfre.seeds.hotpotqa import load as load_hotpotqa

JUDGE_MOD = {
    "claude_sonnet_4_6": claude_judge,
    "mistral_large_2": mistral_judge,
    "faithjudge_style_sonnet": faithjudge,
    "hhem_2_1_open": hhem_judge,
    "minicheck_flan_t5_large": minicheck_judge,
    "ragas_style_sonnet": ragas_judge,
    "alignscore_large": alignscore_judge,
}
# Legacy judges excluded from the paper -- do not bother re-scoring.
SKIP_JUDGES = {"glm_4_7_cerebras", "qwen3_235b_cerebras", "claude_opus_4_7"}


def _free():
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


def build_answer_map(pilot_dir, cited_answers):
    """(seed_id, operator) -> answer text for every scored cell."""
    amap = {}
    # clean / clean_cited answers come from seeds files
    for line in (Path(pilot_dir) / "seeds.jsonl").open():
        s = json.loads(line)
        if not s.get("accepted"):
            continue
        if "cited_answer" in s:  # citation pilot
            amap[(s["seed_id"], "clean_cited")] = s["cited_answer"]
            cited_answers[s["seed_id"]] = s["cited_answer"]
        else:
            amap[(s["seed_id"], "clean")] = s["gold_answer"]
    # perturbed answers
    pf = Path(pilot_dir) / "perturbations.jsonl"
    if pf.exists():
        for line in pf.open():
            p = json.loads(line)
            if p.get("rule_passed"):
                amap[(p["seed_id"], p["operator"])] = p["perturbed_answer"]
    return amap


def corrupted_cells(pilot_dir):
    """(seed,op,judge) that were rescored with empty premise (had error in backup)."""
    baks = sorted(glob.glob(str(Path(pilot_dir) / "verdicts.jsonl.predeudpe-bak-*")))
    if not baks:
        return set()
    out = set()
    for line in Path(baks[-1]).open():
        r = json.loads(line)
        if r.get("metadata", {}).get("error") and r["judge"] not in SKIP_JUDGES:
            out.add((r["seed_id"], r["operator"], r["judge"]))
    return out


def all_alignscore_cells(pilot_dir):
    """Every (seed,op) that currently has an alignscore verdict -> re-score all."""
    out = set()
    for line in (Path(pilot_dir) / "verdicts.jsonl").open():
        r = json.loads(line)
        if r["judge"] == "alignscore_large":
            out.add((r["seed_id"], r["operator"]))
    return out


def verdict_row(v):
    return {
        "seed_id": v.seed_id, "operator": v.operator, "judge": v.judge_name,
        "verdict": v.verdict, "reasoning": v.judge_metadata.get("reasoning"),
        "metadata": {k: v.judge_metadata[k] for k in ("model", "error") if k in v.judge_metadata},
    }


def main():
    smap = build_seed_map()
    cited_answers = {}

    for pilot_dir in ("results/preview_pilot", "results/citation_relocation_pilot"):
        amap = build_answer_map(pilot_dir, cited_answers)
        out_file = Path(pilot_dir) / "verdicts_refixed.jsonl"

        done = set()
        if out_file.exists():
            for line in out_file.open():
                r = json.loads(line)
                done.add((r["seed_id"], r["operator"], r["judge"]))

        # Work items: all AlignScore cells (re-score wholesale) + corrupted others
        work = []
        for (sid, op) in all_alignscore_cells(pilot_dir):
            work.append((sid, op, "alignscore_large"))
        for (sid, op, judge) in corrupted_cells(pilot_dir):
            if judge != "alignscore_large":
                work.append((sid, op, judge))
        work = [w for w in work if w not in done]

        print(f"\n[{pilot_dir}] {len(work)} cells to re-score "
              f"(device={alignscore_judge._select_device()})", flush=True)

        with out_file.open("a") as f:
            for i, (sid, op, judge) in enumerate(work, 1):
                seed = smap.get(sid)
                answer = amap.get((sid, op))
                if seed is None or answer is None:
                    f.write(json.dumps({"seed_id": sid, "operator": op, "judge": judge,
                                        "verdict": "error", "reasoning": None,
                                        "metadata": {"error": "missing seed/answer in refix"}}) + "\n")
                    continue
                # citation cells need cited_answer in metadata for some judges
                if op in ("citation_relocation", "clean_cited"):
                    seed.metadata["cited_answer"] = cited_answers.get(sid, "")
                try:
                    v = JUDGE_MOD[judge].score(seed, answer, operator=op)
                    f.write(json.dumps(verdict_row(v)) + "\n")
                except Exception as e:
                    f.write(json.dumps({"seed_id": sid, "operator": op, "judge": judge,
                                        "verdict": "error", "reasoning": None,
                                        "metadata": {"error": str(e)[:160]}}) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"  {i}/{len(work)} re-scored", flush=True)
                if judge == "alignscore_large":
                    _free()

        print(f"[{pilot_dir}] wrote {out_file}", flush=True)

    print("\nDone. Next: scripts/rebuild_verdicts.py to merge refixed verdicts.", flush=True)


if __name__ == "__main__":
    main()
