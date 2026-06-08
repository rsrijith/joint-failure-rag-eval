"""Full data-integrity audit of both pilot outputs.

Reports: verdict counts, error rates per judge & operator, cell coverage,
duplicate detection, Phase 1 / Phase 2 consistency, cache integrity.

Run from repo root:
    python scripts/full_audit.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEADLINE_JUDGES = {
    'claude_sonnet_4_6', 'mistral_large_2', 'hhem_2_1_open',
    'minicheck_flan_t5_large', 'alignscore_large', 'ragas_style_sonnet',
    'faithjudge_style_sonnet',
}

LEGACY_JUDGES = {'glm_4_7_cerebras', 'qwen3_235b_cerebras', 'claude_opus_4_7'}


def audit_pilot(name: str, base: Path):
    print(f"\n{'='*70}")
    print(f"  PILOT: {name}  ({base})")
    print(f"{'='*70}\n")

    seeds_file = base / 'seeds.jsonl'
    perts_file = base / 'perturbations.jsonl'
    verdicts_file = base / 'verdicts.jsonl'

    # === 1. Seeds.jsonl integrity ===
    seeds = []
    seed_ids = Counter()
    accepted_seeds = 0
    if seeds_file.exists():
        for line in seeds_file.open():
            r = json.loads(line)
            seeds.append(r)
            seed_ids[r['seed_id']] += 1
            if r.get('accepted'):
                accepted_seeds += 1

    seed_dupes = {sid: c for sid, c in seed_ids.items() if c > 1}
    print(f"[1] SEEDS")
    print(f"    Records: {len(seeds)}")
    print(f"    Distinct seed_ids: {len(seed_ids)}")
    print(f"    Duplicates: {len(seed_dupes)}" + (f"  ({list(seed_dupes.items())[:3]})" if seed_dupes else ""))
    print(f"    ACCEPTED (passed Phase 1 filter): {accepted_seeds}")
    print(f"    Rejected: {len(seeds) - accepted_seeds}")

    # === 2. Perturbations.jsonl integrity ===
    perts = []
    pert_keys = Counter()
    pert_by_op = Counter()
    rule_failed = Counter()
    if perts_file.exists():
        for line in perts_file.open():
            r = json.loads(line)
            perts.append(r)
            pert_keys[(r['seed_id'], r['operator'])] += 1
            pert_by_op[r['operator']] += 1
            if not r.get('rule_passed'):
                rule_failed[r['operator']] += 1

    pert_dupes = {k: c for k, c in pert_keys.items() if c > 1}
    print(f"\n[2] PERTURBATIONS")
    print(f"    Records: {len(perts)}")
    print(f"    Distinct (seed, op) keys: {len(pert_keys)}")
    print(f"    Duplicates: {len(pert_dupes)}" + (f"  ({list(pert_dupes.items())[:3]})" if pert_dupes else ""))
    print(f"    By operator:")
    for op, c in sorted(pert_by_op.items()):
        rf = rule_failed.get(op, 0)
        print(f"      {op:25s}  total={c}  rule_failed={rf}  rule_passed={c-rf}")

    # === 3. Verdicts.jsonl integrity ===
    verdict_keys = Counter()
    verdict_status = Counter()
    invalid_verdicts = []
    verdict_judges = Counter()
    errors_by_judge_op = defaultdict(lambda: defaultdict(int))
    if verdicts_file.exists():
        for line in verdicts_file.open():
            r = json.loads(line)
            key = (r['seed_id'], r['operator'], r['judge'])
            verdict_keys[key] += 1
            verdict_status[r['verdict']] += 1
            verdict_judges[r['judge']] += 1
            if r['verdict'] not in ('faithful', 'unfaithful', 'error'):
                invalid_verdicts.append(key)
            if r.get('metadata', {}).get('error'):
                errors_by_judge_op[r['judge']][r['operator']] += 1

    verdict_dupes = {k: c for k, c in verdict_keys.items() if c > 1}
    print(f"\n[3] VERDICTS")
    print(f"    Records: {sum(verdict_keys.values())}")
    print(f"    Distinct (seed, op, judge) keys: {len(verdict_keys)}")
    print(f"    Duplicates: {len(verdict_dupes)}" + (f"  (first 3: {list(verdict_dupes.items())[:3]})" if verdict_dupes else ""))
    print(f"    Verdict values:")
    for v, c in sorted(verdict_status.items()):
        print(f"      {v:15s} {c}")
    if invalid_verdicts:
        print(f"    INVALID verdict values: {len(invalid_verdicts)}")
    print(f"    Judges seen:")
    for j, c in sorted(verdict_judges.items()):
        legacy_tag = '  (LEGACY)' if j in LEGACY_JUDGES else ''
        print(f"      {j:30s} {c}{legacy_tag}")

    # === 4. ERROR ANALYSIS per judge ===
    print(f"\n[4] ERROR ANALYSIS (verdicts with metadata.error)")
    print(f"    {'Judge':<32} {'Errors':<10} {'TotalCalls':<12} {'ErrRate':<10}")
    for j in sorted(verdict_judges.keys()):
        total = verdict_judges[j]
        errs = sum(errors_by_judge_op[j].values())
        rate = 100 * errs / total if total else 0
        flag = '  *** HIGH ***' if rate > 10 and j in HEADLINE_JUDGES else ''
        print(f"    {j:<32} {errs:<10} {total:<12} {rate:.1f}%{flag}")

    # === 5. PHASE 1 / PHASE 2 CONSISTENCY ===
    print(f"\n[5] PHASE 1 / PHASE 2 CONSISTENCY")
    accepted_set = {s['seed_id'] for s in seeds if s.get('accepted')}
    pert_seed_set = {p['seed_id'] for p in perts}
    accepted_no_pert = accepted_set - pert_seed_set
    pert_not_accepted = pert_seed_set - accepted_set
    print(f"    Accepted seeds without perturbations: {len(accepted_no_pert)}" +
          (f"  (first 3: {list(accepted_no_pert)[:3]})" if accepted_no_pert else ""))
    print(f"    Perturbations for non-accepted seeds: {len(pert_not_accepted)}" +
          (f"  (first 3: {list(pert_not_accepted)[:3]})" if pert_not_accepted else ""))

    # === 6. CELL COVERAGE (per accepted seed × rule-passed op, do we have all 7 judges?) ===
    cells = defaultdict(set)  # (seed, op) -> set of judges with non-error verdict
    cells_with_any = defaultdict(set)  # incl error verdicts
    for line in verdicts_file.open() if verdicts_file.exists() else []:
        r = json.loads(line)
        key = (r['seed_id'], r['operator'])
        cells_with_any[key].add(r['judge'])
        if not r.get('metadata', {}).get('error') and r['judge'] in HEADLINE_JUDGES:
            cells[key].add(r['judge'])

    # Expected cells: each accepted seed × each rule-passed op (where op != 'clean')
    expected_cells = set()
    for p in perts:
        if p.get('rule_passed') and p['seed_id'] in accepted_set:
            expected_cells.add((p['seed_id'], p['operator']))

    full_cells = sum(1 for k in expected_cells if len(cells[k]) >= 7)
    partial_cells = sum(1 for k in expected_cells if 0 < len(cells[k]) < 7)
    missing_cells = sum(1 for k in expected_cells if len(cells[k]) == 0)

    print(f"\n[6] CELL COVERAGE (accepted seeds × rule-passed operators, 7 headline judges)")
    print(f"    Expected cells: {len(expected_cells)}")
    print(f"    Fully scored (all 7 judges clean): {full_cells}")
    print(f"    Partially scored (some clean, some missing/errored): {partial_cells}")
    print(f"    Zero clean scores: {missing_cells}")

    # By op
    by_op_full = defaultdict(int)
    by_op_partial = defaultdict(int)
    for k in expected_cells:
        sid, op = k
        if len(cells[k]) >= 7:
            by_op_full[op] += 1
        elif len(cells[k]) > 0:
            by_op_partial[op] += 1
    print(f"    By operator (full / partial):")
    for op in sorted(set(list(by_op_full.keys()) + list(by_op_partial.keys()))):
        print(f"      {op:25s}  full={by_op_full[op]}  partial={by_op_partial[op]}")

    # === 7. CLEAN PHASE COVERAGE (for completeness) ===
    clean_cells = defaultdict(set)
    for k, judges in cells.items():
        if k[1] in ('clean', 'clean_cited'):
            clean_cells[k[0]] = judges
    clean_op = 'clean_cited' if name == 'citation_relocation' else 'clean'
    clean_full = sum(1 for s in accepted_set if len(clean_cells.get(s, set())) >= 7)
    print(f"\n[7] CLEAN PHASE COVERAGE (accepted seeds with all 7 clean verdicts)")
    print(f"    Accepted seeds: {len(accepted_set)}")
    print(f"    With all 7 clean verdicts: {clean_full}")


def audit_cited_cache():
    print(f"\n{'='*70}")
    print(f"  CITED-ANNOTATION CACHE")
    print(f"{'='*70}\n")
    p = Path('data/cache/expertqa_cited.jsonl')
    if not p.exists():
        print("  (cache file does not exist)")
        return

    total = 0
    seen = set()
    dupes = []
    errors = 0
    empty = 0
    usable = 0
    unusable_single = 0
    for line in p.open():
        r = json.loads(line)
        total += 1
        sid = r['seed_id']
        if sid in seen:
            dupes.append(sid)
            continue
        seen.add(sid)
        if r.get('error'):
            errors += 1
        elif not r.get('cited_answer'):
            empty += 1
        elif len(r.get('distinct_indices', [])) < 2:
            unusable_single += 1
        else:
            usable += 1

    print(f"  Total cache lines: {total}")
    print(f"  Distinct seed_ids: {len(seen)}")
    print(f"  Duplicates: {len(dupes)}" + (f"  ({dupes[:3]})" if dupes else ""))
    print(f"  Errored: {errors}")
    print(f"  Empty cited_answer: {empty}")
    print(f"  Has cited but <2 distinct citations (unusable for citation_relocation): {unusable_single}")
    print(f"  Usable: {usable}")


def main():
    audit_pilot("main_preview", Path('results/preview_pilot'))
    audit_pilot("citation_relocation", Path('results/citation_relocation_pilot'))
    audit_cited_cache()


if __name__ == '__main__':
    main()
