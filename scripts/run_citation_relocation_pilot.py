"""Citation-relocation pilot (6th operator, ExpertQA only).

Runs the citation_relocation perturbation on Claude-annotated ExpertQA
seeds. Independent of the main 5-operator pilot — separate output dir,
separate seeds, separate verdicts. Combined analysis reads from both.

Mechanism:
  Phase 1 (per seed):
    1. Claude annotates gold_answer with [N] passage-index citations
       (cached via expertqa_cited loader).
    2. Skip seeds whose annotation yields < 2 distinct citations.
    3. Score the CITED clean answer with all 8 judges. This is the
       per-seed "is the cited answer faithful" baseline. Seeds where
       fewer than 4/8 judges call the cited_answer faithful are
       dropped from the perturbation phase (the operator's failure
       signal only makes sense from a faithful starting point).

  Phase 2 (per accepted seed):
    1. Apply citation_relocation: swap [N] indices via a non-identity
       permutation. Each claim is now mis-attributed to a passage that
       does NOT support it (though the claim itself is still supported
       by SOME passage in the context).
    2. Score the perturbed answer with all 8 judges.
    3. Joint failure = all 8 judges call the perturbed (mis-attributed)
       answer faithful.

Target: 100 accepted-and-perturbed seeds.

Resumability: same append-only JSONL pattern as run_preview_pilot.py.
Cache key for verdicts is (seed_id, operator, judge_name); skip-on-error
means failed judge calls are NOT cached and will be retried on next run.

Outputs:
    results/citation_relocation_pilot/seeds.jsonl
    results/citation_relocation_pilot/perturbations.jsonl
    results/citation_relocation_pilot/verdicts.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jfre.judges import (
    alignscore_judge,
    claude_judge,
    faithjudge,
    glm_cerebras_judge,
    hhem_judge,
    minicheck_judge,
    mistral_judge,
    ragas_judge,
)
from jfre.operators import citation_relocation
from jfre.seeds.expertqa_cited import load as load_expertqa_cited


N_TARGET_SEEDS = 100
SEED_FAITHFUL_THRESHOLD = 4  # >=4/8 judges must call cited clean faithful

JUDGES = [
    ("claude",      claude_judge),
    ("mistral",     mistral_judge),
    ("hhem",        hhem_judge),
    ("glm",         glm_cerebras_judge),
    ("minicheck",   minicheck_judge),
    ("alignscore",  alignscore_judge),
    ("ragas",       ragas_judge),
    ("faithjudge",  faithjudge),
]

OUT_DIR = Path("results/citation_relocation_pilot")
SEEDS_FILE = OUT_DIR / "seeds.jsonl"
PERTURBATIONS_FILE = OUT_DIR / "perturbations.jsonl"
VERDICTS_FILE = OUT_DIR / "verdicts.jsonl"


def _verdict_to_dict(v) -> dict:
    return {
        "seed_id": v.seed_id,
        "operator": v.operator,
        "judge": v.judge_name,
        "verdict": v.verdict,
        "reasoning": v.judge_metadata.get("reasoning"),
        "metadata": {k: v.judge_metadata[k] for k in ("model", "error") if k in v.judge_metadata},
    }


def _perturbation_to_dict(p, seed) -> dict:
    return {
        "seed_id": p.seed_id,
        "source": seed.source,
        "operator": p.operator,
        "rule_passed": p.rule_passed,
        "rule_notes": p.rule_notes,
        "perturbed_answer": p.perturbed_answer,
        "edit_diff": p.edit_diff,
        "question": seed.question,
        "gold_answer": seed.gold_answer,
        "cited_answer": seed.metadata.get("cited_answer", ""),
    }


def _load_done_verdicts() -> set[tuple]:
    if not VERDICTS_FILE.exists():
        return set()
    done: set[tuple] = set()
    for line in VERDICTS_FILE.open():
        r = json.loads(line)
        done.add((r["seed_id"], r["operator"], r["judge"]))
    return done


def _load_done_perturbations() -> dict[tuple, dict]:
    if not PERTURBATIONS_FILE.exists():
        return {}
    done: dict[tuple, dict] = {}
    for line in PERTURBATIONS_FILE.open():
        r = json.loads(line)
        done[(r["seed_id"], r["operator"])] = r
    return done


def _load_done_seeds() -> dict[str, dict]:
    if not SEEDS_FILE.exists():
        return {}
    done: dict[str, dict] = {}
    for line in SEEDS_FILE.open():
        r = json.loads(line)
        done[r["seed_id"]] = r
    return done


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done_verdicts = _load_done_verdicts()
    done_perturbations = _load_done_perturbations()
    done_seeds = _load_done_seeds()
    print(f"Resuming. Already done: {len(done_seeds)} seeds filtered, "
          f"{len(done_perturbations)} perturbations, {len(done_verdicts)} verdicts.\n")

    # ---- Phase 1: load Claude-cited seeds, score the cited clean answer.
    accepted_seeds: list = []
    current_judge_names = {mod.JUDGE_NAME for _name, mod in JUDGES}

    # Pull more raw than the target since not every cited_answer will be
    # judged faithful by 4+/8 judges.
    raw_iter = load_expertqa_cited(n=N_TARGET_SEEDS * 3)

    with SEEDS_FILE.open("a") as seeds_f, VERDICTS_FILE.open("a") as verdicts_f:
        for i, seed in enumerate(raw_iter):
            if len(accepted_seeds) >= N_TARGET_SEEDS:
                break

            cited = seed.metadata.get("cited_answer", "")
            if not cited:
                continue

            if seed.seed_id in done_seeds:
                rec = done_seeds[seed.seed_id]
                if rec["accepted"]:
                    accepted_seeds.append(seed)
                    print(f"  [{i+1:3d}] {seed.seed_id} (cached) {rec['n_judges_faithful']}/8 -> ACCEPT")
                else:
                    print(f"  [{i+1:3d}] {seed.seed_id} (cached) {rec['n_judges_faithful']}/8 -> reject")
                continue

            # Fresh: score the CITED clean answer with each judge.
            for _name, mod in JUDGES:
                key = (seed.seed_id, "clean_cited", mod.JUDGE_NAME)
                if key in done_verdicts:
                    continue
                try:
                    v = mod.score(seed, cited, operator="clean_cited")
                except Exception as e:
                    print(f"     [skip {mod.JUDGE_NAME} on clean_cited]: {str(e)[:120]}")
                    continue
                verdicts_f.write(json.dumps(_verdict_to_dict(v)) + "\n")
                verdicts_f.flush()
                done_verdicts.add(key)

            # Tally clean_cited verdicts from the current judge set only.
            all_clean = [
                json.loads(line) for line in VERDICTS_FILE.open()
                if json.loads(line)["seed_id"] == seed.seed_id
                and json.loads(line)["operator"] == "clean_cited"
                and json.loads(line)["judge"] in current_judge_names
            ]
            n_faithful = sum(1 for r in all_clean if r["verdict"] == "faithful")
            accepted = n_faithful >= SEED_FAITHFUL_THRESHOLD

            seed_rec = {
                "seed_id": seed.seed_id,
                "source": seed.source,
                "question": seed.question,
                "gold_answer": seed.gold_answer,
                "cited_answer": cited,
                "distinct_citation_indices": seed.metadata.get("distinct_citation_indices", []),
                "n_judges_faithful": n_faithful,
                "accepted": accepted,
            }
            seeds_f.write(json.dumps(seed_rec) + "\n")
            seeds_f.flush()
            done_seeds[seed.seed_id] = seed_rec
            print(f"  [{i+1:3d}] {seed.seed_id} {n_faithful}/8 cited-faithful  {'ACCEPT' if accepted else 'reject'}")
            if accepted:
                accepted_seeds.append(seed)

    print(f"\nPhase 1 complete: {len(accepted_seeds)} accepted cited seeds.\n")

    # ---- Phase 2: relocate citations, score perturbed answer with 8 judges.
    print(f"Phase 2: apply citation_relocation + score on {len(accepted_seeds)} seeds.\n")

    with PERTURBATIONS_FILE.open("a") as p_f, VERDICTS_FILE.open("a") as v_f:
        for i, seed in enumerate(accepted_seeds):
            print(f"[{i+1:3d}/{len(accepted_seeds)}] {seed.seed_id}")

            pert_key = (seed.seed_id, "citation_relocation")
            if pert_key in done_perturbations:
                pert_rec = done_perturbations[pert_key]
                rule_passed = pert_rec["rule_passed"]
                perturbed_answer = pert_rec["perturbed_answer"]
                cached_marker = " (cached)"
            else:
                pert = citation_relocation.generate(seed)
                rule_passed = pert.rule_passed
                perturbed_answer = pert.perturbed_answer
                p_f.write(json.dumps(_perturbation_to_dict(pert, seed)) + "\n")
                p_f.flush()
                done_perturbations[pert_key] = _perturbation_to_dict(pert, seed)
                cached_marker = ""

            if not rule_passed:
                notes = done_perturbations[pert_key]["rule_notes"][:80]
                print(f"     citation_relocation{cached_marker} SKIP  ({notes})")
                continue

            for jname, jmod in JUDGES:
                key = (seed.seed_id, "citation_relocation", jmod.JUDGE_NAME)
                if key in done_verdicts:
                    continue
                try:
                    v = jmod.score(seed, perturbed_answer, operator="citation_relocation")
                except Exception as e:
                    print(f"     [skip {jmod.JUDGE_NAME} on citation_relocation]: {str(e)[:120]}")
                    continue
                v_f.write(json.dumps(_verdict_to_dict(v)) + "\n")
                done_verdicts.add(key)
            v_f.flush()

            all_pert_verdicts = [
                json.loads(line) for line in VERDICTS_FILE.open()
                if json.loads(line)["seed_id"] == seed.seed_id
                and json.loads(line)["operator"] == "citation_relocation"
                and json.loads(line)["judge"] in current_judge_names
            ]
            n_pert_faithful = sum(1 for r in all_pert_verdicts if r["verdict"] == "faithful")
            joint = " <-- JOINT FAILURE" if n_pert_faithful == len(JUDGES) else ""
            summary = " ".join(f"{r['judge'][:6]}={r['verdict'][:5]}" for r in all_pert_verdicts)
            print(f"     citation_relocation{cached_marker} {n_pert_faithful}/8: {summary}{joint}")

    print(f"\nDone. Outputs:")
    print(f"  {SEEDS_FILE}")
    print(f"  {PERTURBATIONS_FILE}")
    print(f"  {VERDICTS_FILE}")


if __name__ == "__main__":
    sys.exit(main())
