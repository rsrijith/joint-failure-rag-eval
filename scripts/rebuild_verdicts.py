"""Merge refixed verdicts into corrected verdicts files.

For each pilot, replaces every (seed, op, judge) verdict that appears in
verdicts_refixed.jsonl with the refixed version, keeping all other verdicts
unchanged. Backs up the current verdicts.jsonl first.

Run AFTER scripts/refix_corrupted_verdicts.py finishes.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def main():
    for pilot_dir in ("results/preview_pilot", "results/citation_relocation_pilot"):
        vfile = Path(pilot_dir) / "verdicts.jsonl"
        rfile = Path(pilot_dir) / "verdicts_refixed.jsonl"
        if not rfile.exists():
            print(f"[{pilot_dir}] no refixed file; skipping")
            continue

        # Load refixed verdicts (latest wins per key)
        refixed = {}
        n_refix_err = 0
        for line in rfile.open():
            r = json.loads(line)
            key = (r["seed_id"], r["operator"], r["judge"])
            refixed[key] = r
            if r.get("metadata", {}).get("error"):
                n_refix_err += 1

        # Back up current verdicts
        ts = int(time.time())
        bak = vfile.with_suffix(f".jsonl.precorrect-bak-{ts}")
        shutil.copy(vfile, bak)

        # Rewrite: replace any key present in refixed
        kept, replaced = 0, 0
        out_lines = []
        for line in vfile.open():
            r = json.loads(line)
            key = (r["seed_id"], r["operator"], r["judge"])
            if key in refixed:
                out_lines.append(json.dumps(refixed[key]))
                replaced += 1
            else:
                out_lines.append(json.dumps(r))
                kept += 1

        # Any refixed keys not already in verdicts? (shouldn't happen, but append)
        existing_keys = set()
        for line in vfile.open():
            r = json.loads(line)
            existing_keys.add((r["seed_id"], r["operator"], r["judge"]))
        added = 0
        for key, r in refixed.items():
            if key not in existing_keys:
                out_lines.append(json.dumps(r))
                added += 1

        vfile.write_text("\n".join(out_lines) + "\n")
        print(f"[{pilot_dir}] replaced {replaced}, kept {kept}, added {added}, "
              f"refix-errors {n_refix_err}. backup: {bak.name}")

    print("\nDone. Re-run analyses: full_audit, corrected_headline_analysis, "
          "combined_analysis, hypothesis_check_v2.")


if __name__ == "__main__":
    main()
