"""Run the Day 5 preview pilot.

50 HotpotQA seeds × 2 operators (entity_swap, numeric_drift) × 3 judges
(claude, qwen, mistral). Saves all verdicts and perturbations as JSONL
for analysis by scripts/analyze_pilot.py.

This is the "preview" pilot — full methodology calls for 8 judges and
6 operators. Here we run a subset to validate orchestration and produce
first joint-failure numbers before adding the local NLI judges.

Output:
    results/preview_pilot/seeds.jsonl
    results/preview_pilot/perturbations.jsonl
    results/preview_pilot/verdicts.jsonl

Run from repo root:
    python scripts/run_preview_pilot.py
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
from jfre.operators import (
    distractor_parroting,
    entity_swap,
    hedge_insertion,
    numeric_drift,
    paraphrase_null,
)
from jfre.seeds.expertqa import load as load_expertqa
from jfre.seeds.hotpotqa import load as load_hotpotqa


# Per-source seed targets and raw pool sizes.
# Methodology target: 500 accepted seeds, balanced across sources.
# Resumability: previously-accepted seeds (50 HotpotQA + 25 ExpertQA from
# preview pilot) reuse cached decisions; only new raw seeds get fresh-filtered.
SOURCE_TARGETS: list[tuple[str, int, int, object]] = [
    ("hotpotqa", 250, 1000, load_hotpotqa),
    ("expertqa", 250, 1500, load_expertqa),
]
N_SEEDS = sum(t[1] for t in SOURCE_TARGETS)  # 500 total accepted
RAW_POOL = sum(t[2] for t in SOURCE_TARGETS)  # legacy field; unused below

OPERATORS = [
    ("entity_swap",          entity_swap),
    ("numeric_drift",        numeric_drift),
    ("hedge_insertion",      hedge_insertion),
    ("distractor_parroting", distractor_parroting),
    ("paraphrase_null",      paraphrase_null),
]

# Full 8-judge ensemble:
#   3 NLI/fact-checkers: HHEM-2.1-Open, MiniCheck-Flan-T5-L, AlignScore-large
#   1 claim-decomposition: RAGAS-style (Claude Sonnet backing)
#   4 LLM-as-judges:
#     - Claude 4 (Anthropic)
#     - Mistral Large 2 (Mistral)
#     - GLM-4.7 via Cerebras (Z.AI)
#     - FaithJudge-style (Claude Sonnet with Vectara hallucination few-shot)
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

# Pre-filter threshold. With 5 judges, 4/5 = 80% mirrors the methodology's
# 7/8 = 87.5% (closest match without being 100%). Cached HotpotQA seeds from
# the earlier 3-of-3 ensemble are stricter, so their accepted flags still hold.
SEED_FAITHFUL_THRESHOLD = 4

OUT_DIR = Path("results/preview_pilot")
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
    }


def _load_done_verdicts() -> set[tuple]:
    """(seed_id, operator, judge_name) tuples for verdicts already persisted."""
    if not VERDICTS_FILE.exists():
        return set()
    done: set[tuple] = set()
    for line in VERDICTS_FILE.open():
        r = json.loads(line)
        done.add((r["seed_id"], r["operator"], r["judge"]))
    return done


def _load_done_perturbations() -> dict[tuple, dict]:
    """(seed_id, operator) -> perturbation record for perturbations already persisted."""
    if not PERTURBATIONS_FILE.exists():
        return {}
    done: dict[tuple, dict] = {}
    for line in PERTURBATIONS_FILE.open():
        r = json.loads(line)
        done[(r["seed_id"], r["operator"])] = r
    return done


def _load_done_seeds() -> dict[str, dict]:
    """seed_id -> seed log entry for seeds already filtered."""
    if not SEEDS_FILE.exists():
        return {}
    done: dict[str, dict] = {}
    for line in SEEDS_FILE.open():
        r = json.loads(line)
        done[r["seed_id"]] = r
    return done


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Resumability: load any already-persisted state.
    done_verdicts = _load_done_verdicts()
    done_perturbations = _load_done_perturbations()
    done_seeds = _load_done_seeds()
    print(f"Resuming. Already done: {len(done_seeds)} seeds filtered, "
          f"{len(done_perturbations)} perturbations, {len(done_verdicts)} judge verdicts.\n")

    # ---- Phase 1: per-source seed-faithful pre-filter (with resumability)
    accepted_seeds: list = []
    current_judge_names = {mod.JUDGE_NAME for _name, mod in JUDGES}

    for source_name, target, pool, loader in SOURCE_TARGETS:
        print(f"\nPhase 1 [{source_name}]: target {target} accepted from a pool of {pool} raw")
        raw_seeds = list(loader(n=pool))
        print(f"Loaded {len(raw_seeds)} raw {source_name} seeds.")

        source_accepted = 0
        with SEEDS_FILE.open("a") as seeds_f, VERDICTS_FILE.open("a") as verdicts_f:
            for i, seed in enumerate(raw_seeds):
                if source_accepted >= target:
                    break

                # If cached, reuse decision
                if seed.seed_id in done_seeds:
                    rec = done_seeds[seed.seed_id]
                    if rec["accepted"]:
                        accepted_seeds.append(seed)
                        source_accepted += 1
                        print(f"  [{i+1:3d}] {seed.seed_id[:35]:35s} (cached) {rec['n_judges_faithful']}/{len(JUDGES)} -> ACCEPT")
                    else:
                        print(f"  [{i+1:3d}] {seed.seed_id[:35]:35s} (cached) {rec['n_judges_faithful']}/{len(JUDGES)} -> reject")
                    continue

                # Fresh: score clean gold with each judge not already cached
                for _name, mod in JUDGES:
                    key = (seed.seed_id, "clean", mod.JUDGE_NAME)
                    if key in done_verdicts:
                        continue
                    try:
                        v = mod.score(seed, seed.gold_answer, operator="clean")
                    except Exception as e:
                        # Single-judge failure must not crash the pilot. We do NOT
                        # write a verdict and do NOT add to done_verdicts, so a
                        # later resumed session re-tries this judge for this seed.
                        print(f"     [skip {mod.JUDGE_NAME} on clean]: {str(e)[:120]}")
                        continue
                    verdicts_f.write(json.dumps(_verdict_to_dict(v)) + "\n")
                    verdicts_f.flush()
                    done_verdicts.add(key)

                # Tally clean verdicts (only from current judge set)
                all_clean = [
                    json.loads(line) for line in VERDICTS_FILE.open()
                    if json.loads(line)["seed_id"] == seed.seed_id
                    and json.loads(line)["operator"] == "clean"
                    and json.loads(line)["judge"] in current_judge_names
                ]
                n_faithful = sum(1 for r in all_clean if r["verdict"] == "faithful")
                accepted = n_faithful >= SEED_FAITHFUL_THRESHOLD

                seed_rec = {
                    "seed_id": seed.seed_id,
                    "source": seed.source,
                    "question": seed.question,
                    "gold_answer": seed.gold_answer,
                    "metadata": seed.metadata,
                    "n_judges_faithful": n_faithful,
                    "accepted": accepted,
                }
                seeds_f.write(json.dumps(seed_rec) + "\n")
                seeds_f.flush()
                done_seeds[seed.seed_id] = seed_rec

                print(f"  [{i+1:3d}] {seed.seed_id[:35]:35s} {n_faithful}/{len(JUDGES)} faithful  {'ACCEPT' if accepted else 'reject'}")
                if accepted:
                    accepted_seeds.append(seed)
                    source_accepted += 1

        print(f"[{source_name}] phase 1: accepted {source_accepted} (target {target})")

    print(f"\nPhase 1 complete: {len(accepted_seeds)} accepted seeds total.\n")

    # ---- Phase 2: generate perturbations + score with judges (resumable)
    print(f"Phase 2: generate perturbations + run judges on {len(accepted_seeds)} accepted seeds.\n")

    with PERTURBATIONS_FILE.open("a") as p_f, VERDICTS_FILE.open("a") as v_f:
        for i, seed in enumerate(accepted_seeds):
            print(f"[{i+1:2d}/{len(accepted_seeds)}] {seed.seed_id[:35]}")

            for op_name, op_mod in OPERATORS:
                pert_key = (seed.seed_id, op_name)

                # Generate perturbation (skip if already done)
                if pert_key in done_perturbations:
                    pert_rec = done_perturbations[pert_key]
                    rule_passed = pert_rec["rule_passed"]
                    perturbed_answer = pert_rec["perturbed_answer"]
                    edit_diff = pert_rec.get("edit_diff", {})
                    cached_marker = " (cached)"
                else:
                    pert = op_mod.generate(seed)
                    rule_passed = pert.rule_passed
                    perturbed_answer = pert.perturbed_answer
                    edit_diff = pert.edit_diff
                    p_f.write(json.dumps(_perturbation_to_dict(pert, seed)) + "\n")
                    p_f.flush()
                    done_perturbations[pert_key] = _perturbation_to_dict(pert, seed)
                    cached_marker = ""

                if not rule_passed:
                    notes = done_perturbations[pert_key]["rule_notes"][:60]
                    print(f"     {op_name:14s}{cached_marker} SKIP  ({notes})")
                    continue

                # Score with all judges (skip ones already done, skip on error)
                pert_verdict_rows = []
                for jname, jmod in JUDGES:
                    key = (seed.seed_id, op_name, jmod.JUDGE_NAME)
                    if key in done_verdicts:
                        continue
                    try:
                        v = jmod.score(seed, perturbed_answer, operator=op_name)
                    except Exception as e:
                        # Per-judge failure: skip, don't cache, let next session retry.
                        print(f"     [skip {jmod.JUDGE_NAME} on {op_name}]: {str(e)[:120]}")
                        continue
                    pert_verdict_rows.append((jname, v.verdict))
                    v_f.write(json.dumps(_verdict_to_dict(v)) + "\n")
                    done_verdicts.add(key)
                v_f.flush()

                # Recover all current-judge verdicts for joint-failure print
                current_judge_names = {mod.JUDGE_NAME for _name, mod in JUDGES}
                all_pert_verdicts = [
                    json.loads(line) for line in VERDICTS_FILE.open()
                    if json.loads(line)["seed_id"] == seed.seed_id
                    and json.loads(line)["operator"] == op_name
                    and json.loads(line)["judge"] in current_judge_names
                ]
                n_pert_faithful = sum(1 for r in all_pert_verdicts if r["verdict"] == "faithful")
                joint = " <-- JOINT FAILURE" if n_pert_faithful == len(JUDGES) else ""
                summary = " ".join(f"{r['judge'][:6]}={r['verdict'][:5]}" for r in all_pert_verdicts)
                print(f"     {op_name:14s}{cached_marker} {summary}{joint}")

    print(f"\nDone. Outputs:")
    print(f"  {SEEDS_FILE}")
    print(f"  {PERTURBATIONS_FILE}")
    print(f"  {VERDICTS_FILE}")
    print(f"\nNext: python scripts/analyze_preview_pilot.py")


if __name__ == "__main__":
    sys.exit(main())
