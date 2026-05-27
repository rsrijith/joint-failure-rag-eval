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

from jfre.judges import claude_judge, hhem_judge, mistral_judge, qwen_cerebras_judge
from jfre.operators import (
    distractor_parroting,
    entity_swap,
    hedge_insertion,
    numeric_drift,
    paraphrase_null,
)
from jfre.seeds.hotpotqa import load


N_SEEDS = 50
RAW_POOL = 200  # load more than N_SEEDS to allow rejection by the seed-faithful filter

OPERATORS = [
    ("entity_swap",          entity_swap),
    ("numeric_drift",        numeric_drift),
    ("hedge_insertion",      hedge_insertion),
    ("distractor_parroting", distractor_parroting),
    ("paraphrase_null",      paraphrase_null),
]

# 4-judge setup: 1 NLI (HHEM) + 3 LLM-judges from different organizations
# (Anthropic, Alibaba via Cerebras, Mistral). Qwen via Cerebras is throttled
# at 1.5s/call self-imposed to avoid free-tier queue-exceeded errors.
JUDGES = [
    ("claude",  claude_judge),
    ("mistral", mistral_judge),
    ("hhem",    hhem_judge),
    ("qwen",    qwen_cerebras_judge),
]

# Pre-filter threshold: was applied during phase 1 when seeds were filtered.
# Cached "accepted" decisions are reused on resume, so this value only affects
# any *new* seeds processed (none expected since N_SEEDS already reached).
SEED_FAITHFUL_THRESHOLD = 3

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

    print(f"Loading up to {RAW_POOL} raw HotpotQA seeds (need {N_SEEDS} that pass seed-faithful filter)...")
    raw_seeds = list(load(n=RAW_POOL))
    print(f"Loaded {len(raw_seeds)} raw seeds.\n")

    # ---- Phase 1: seed-faithful pre-filter (with resumability)
    print(f"Phase 1: seed-faithful pre-filter (threshold: {SEED_FAITHFUL_THRESHOLD}/3 judges say faithful)\n")
    accepted_seeds: list = []

    # Open append-mode for resumability
    with SEEDS_FILE.open("a") as seeds_f, VERDICTS_FILE.open("a") as verdicts_f:
        for i, seed in enumerate(raw_seeds):
            if len(accepted_seeds) >= N_SEEDS:
                break

            # If we already filtered this seed in a prior run, reuse its decision
            if seed.seed_id in done_seeds:
                rec = done_seeds[seed.seed_id]
                if rec["accepted"]:
                    accepted_seeds.append(seed)
                    print(f"  [{i+1:3d}] {seed.seed_id[:35]:35s} (cached) {rec['n_judges_faithful']}/3 -> ACCEPT")
                else:
                    print(f"  [{i+1:3d}] {seed.seed_id[:35]:35s} (cached) {rec['n_judges_faithful']}/3 -> reject")
                continue

            # Score clean gold answer with all 3 judges (skip ones already done)
            clean_verdicts = []
            for _name, mod in JUDGES:
                key = (seed.seed_id, "clean", mod.JUDGE_NAME)
                if key in done_verdicts:
                    # already have this verdict from a prior partial run; will recover from disk below
                    continue
                v = mod.score(seed, seed.gold_answer, operator="clean")
                clean_verdicts.append(v)
                verdicts_f.write(json.dumps(_verdict_to_dict(v)) + "\n")
                verdicts_f.flush()
                done_verdicts.add(key)

            # Now collect clean verdicts for this seed, restricted to the judges
            # currently in JUDGES (so dropped judges from prior runs don't influence
            # the pre-filter decision).
            current_judge_names = {mod.JUDGE_NAME for _name, mod in JUDGES}
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

            print(f"  [{i+1:3d}] {seed.seed_id[:35]:35s} {n_faithful}/3 faithful  {'ACCEPT' if accepted else 'reject'}")
            if accepted:
                accepted_seeds.append(seed)

    print(f"\nPhase 1 complete: {len(accepted_seeds)} accepted seeds.\n")

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

                # Score with all 3 judges (skip ones already done)
                pert_verdict_rows = []
                for jname, jmod in JUDGES:
                    key = (seed.seed_id, op_name, jmod.JUDGE_NAME)
                    if key in done_verdicts:
                        continue
                    v = jmod.score(seed, perturbed_answer, operator=op_name)
                    pert_verdict_rows.append((jname, v.verdict))
                    v_f.write(json.dumps(_verdict_to_dict(v)) + "\n")
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
