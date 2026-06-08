"""Deduplicate verdicts.jsonl, keeping the LATEST non-error verdict per
(seed_id, operator, judge_name). If all verdicts for a key are errored,
keep the latest errored one.

Backs up the original to verdicts.jsonl.dupe-bak-<ts> before rewriting.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', choices=['preview', 'citation'], default='preview')
    args = parser.parse_args()

    if args.pilot == 'preview':
        verdicts_file = Path('results/preview_pilot/verdicts.jsonl')
    else:
        verdicts_file = Path('results/citation_relocation_pilot/verdicts.jsonl')

    # Read all verdicts, keep best one per key
    best = {}  # (seed, op, judge) -> (record_dict, line_index, has_error)
    raw_count = 0
    for i, line in enumerate(verdicts_file.open()):
        r = json.loads(line)
        key = (r['seed_id'], r['operator'], r['judge'])
        has_err = bool(r.get('metadata', {}).get('error'))
        raw_count += 1

        if key not in best:
            best[key] = (r, i, has_err)
            continue

        existing_r, existing_i, existing_err = best[key]
        # Prefer non-error over error; among same-status, prefer later
        if existing_err and not has_err:
            best[key] = (r, i, has_err)
        elif existing_err == has_err and i > existing_i:
            best[key] = (r, i, has_err)

    print(f"Read {raw_count} verdicts, deduped to {len(best)} unique (seed, op, judge) keys")
    err_remaining = sum(1 for (_, _, e) in best.values() if e)
    print(f"  {err_remaining} entries still have metadata.error after dedup")

    # Backup
    ts = int(time.time())
    bak = verdicts_file.with_suffix(f'.jsonl.predeudpe-bak-{ts}')
    shutil.copy(verdicts_file, bak)
    print(f"Backed up original to {bak}")

    # Write deduped file
    with verdicts_file.open('w') as f:
        for (r, i, e) in best.values():
            f.write(json.dumps(r) + '\n')
    print(f"Wrote {len(best)} deduped verdicts to {verdicts_file}")


if __name__ == '__main__':
    main()
