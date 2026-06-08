"""Rescore verdict cells whose existing verdict has metadata.error.

Reads verdicts.jsonl, finds entries with metadata.error from one of the 7
headline judges, looks up seed + answer-text, re-calls the judge, and
appends a fresh verdict. After all rescoring, deduplicates verdicts.jsonl
keeping the LATEST verdict per (seed_id, operator, judge_name), with a
preference for verdicts that don't have metadata.error.

Usage:
    python scripts/rescore_errored.py [--pilot preview|citation] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from jfre.judges import (
    alignscore_judge,
    claude_judge,
    faithjudge,
    hhem_judge,
    minicheck_judge,
    mistral_judge,
    ragas_judge,
)
from jfre.types import Passage, Seed

JUDGES_BY_NAME = {
    'claude_sonnet_4_6': claude_judge,
    'mistral_large_2': mistral_judge,
    'hhem_2_1_open': hhem_judge,
    'minicheck_flan_t5_large': minicheck_judge,
    'alignscore_large': alignscore_judge,
    'ragas_style_sonnet': ragas_judge,
    'faithjudge_style_sonnet': faithjudge,
}


def _seed_from_record(rec: dict) -> Seed:
    """Reconstruct a Seed object from a seeds.jsonl record."""
    psgs = []
    for p in rec.get('metadata', {}).get('passages', []):
        psgs.append(Passage(
            text=p.get('text', ''),
            is_relevant=p.get('is_relevant', True),
        ))
    return Seed(
        seed_id=rec['seed_id'],
        source=rec.get('source', 'unknown'),
        question=rec.get('question', ''),
        gold_answer=rec.get('gold_answer', ''),
        passages=psgs,
        metadata=rec.get('metadata', {}),
    )


def _verdict_to_dict(v) -> dict:
    md = {k: v.judge_metadata[k] for k in ('model', 'error') if k in v.judge_metadata}
    return {
        'seed_id': v.seed_id,
        'operator': v.operator,
        'judge': v.judge_name,
        'verdict': v.verdict,
        'reasoning': v.judge_metadata.get('reasoning'),
        'metadata': md,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', choices=['preview', 'citation'], default='preview')
    parser.add_argument('--dry-run', action='store_true', help='Only print what would be rescored')
    args = parser.parse_args()

    if args.pilot == 'preview':
        base = Path('results/preview_pilot')
    else:
        base = Path('results/citation_relocation_pilot')

    seeds_file = base / 'seeds.jsonl'
    perts_file = base / 'perturbations.jsonl'
    verdicts_file = base / 'verdicts.jsonl'

    # Load seeds and perturbations
    seeds_by_id = {}
    for line in seeds_file.open():
        r = json.loads(line)
        seeds_by_id[r['seed_id']] = r

    perts_by_key = {}  # (seed_id, op) -> {'perturbed_answer': ..., 'cited_answer': ...}
    if perts_file.exists():
        for line in perts_file.open():
            r = json.loads(line)
            perts_by_key[(r['seed_id'], r['operator'])] = r

    # Find errored verdicts to rescore
    to_rescore = []  # list of (seed_id, op, judge_name)
    for line in verdicts_file.open():
        v = json.loads(line)
        if v.get('metadata', {}).get('error') and v['judge'] in JUDGES_BY_NAME:
            to_rescore.append((v['seed_id'], v['operator'], v['judge']))

    print(f"Found {len(to_rescore)} errored verdicts to rescore")
    if args.dry_run:
        from collections import Counter
        by_judge = Counter(j for (_, _, j) in to_rescore)
        for j, c in sorted(by_judge.items()):
            print(f"  {j}: {c}")
        return

    # Rescore each one, appending new verdicts to verdicts.jsonl
    n_done = 0
    n_skipped = 0
    n_still_error = 0
    with verdicts_file.open('a') as f:
        for sid, op, judge_name in to_rescore:
            if sid not in seeds_by_id:
                n_skipped += 1
                continue
            seed = _seed_from_record(seeds_by_id[sid])

            # Determine answer text
            if op == 'clean':
                answer = seed.gold_answer
            elif op == 'clean_cited':
                answer = seeds_by_id[sid].get('cited_answer', '')
                seed.metadata['cited_answer'] = answer
            else:
                pkey = (sid, op)
                if pkey not in perts_by_key:
                    n_skipped += 1
                    continue
                answer = perts_by_key[pkey].get('perturbed_answer', '')
                # For citation_relocation, also set cited_answer in metadata
                if op == 'citation_relocation':
                    seed.metadata['cited_answer'] = perts_by_key[pkey].get('cited_answer', '')

            if not answer:
                n_skipped += 1
                continue

            jmod = JUDGES_BY_NAME[judge_name]
            try:
                v = jmod.score(seed, answer, operator=op)
            except Exception as e:
                print(f"  [HARD FAIL {judge_name} on {sid}/{op}]: {str(e)[:120]}")
                n_skipped += 1
                continue

            vd = _verdict_to_dict(v)
            f.write(json.dumps(vd) + '\n')
            f.flush()
            n_done += 1
            if vd.get('metadata', {}).get('error'):
                n_still_error += 1
            if n_done % 50 == 0:
                print(f"  ... rescored {n_done}/{len(to_rescore)} ({n_still_error} still errored)")

    print(f"\nRescored: {n_done}")
    print(f"Still errored after retry: {n_still_error}")
    print(f"Skipped (missing seed/pert): {n_skipped}")
    print(f"\nVerdicts file now has duplicates (old errored + new clean).")
    print(f"Run: python scripts/dedupe_verdicts.py --pilot {args.pilot}")


if __name__ == '__main__':
    sys.exit(main() or 0)
