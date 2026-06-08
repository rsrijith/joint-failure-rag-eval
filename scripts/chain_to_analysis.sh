#!/bin/bash
# Chain everything: wait for citation pilot exit, then run clean pass,
# dedupe, re-audit, and final analysis. No human in the loop.
#
# Usage:  bash scripts/chain_to_analysis.sh <citation_pilot_pid>

set -u
cd "$(dirname "$0")/.."

CR_PID="${1:?need citation pilot PID as arg}"

echo "[chain-analysis] $(date) waiting for citation pilot PID $CR_PID"
while kill -0 "$CR_PID" 2>/dev/null; do
    sleep 30
done
echo "[chain-analysis] $(date) citation pilot exited"

source .venv/bin/activate
set -a; source .env; set +a

# 1. Clean pass on main pilot (mostly AlignScore re-runs, free local).
echo "[chain-analysis] $(date) rescoring errored cells in MAIN pilot"
python -u scripts/rescore_errored.py --pilot preview > logs/rescore_main.log 2>&1
echo "[chain-analysis] $(date) main rescoring done"

# 2. Clean pass on citation pilot.
echo "[chain-analysis] $(date) rescoring errored cells in CITATION pilot"
python -u scripts/rescore_errored.py --pilot citation > logs/rescore_citation.log 2>&1
echo "[chain-analysis] $(date) citation rescoring done"

# 3. Dedupe verdicts.jsonl in both pilots (keep latest non-error per key).
echo "[chain-analysis] $(date) deduping main pilot verdicts"
python -u scripts/dedupe_verdicts.py --pilot preview > logs/dedupe_main.log 2>&1
echo "[chain-analysis] $(date) deduping citation pilot verdicts"
python -u scripts/dedupe_verdicts.py --pilot citation > logs/dedupe_citation.log 2>&1

# 4. Final integrity audit on the cleaned data.
echo "[chain-analysis] $(date) running full audit on cleaned data"
python -u scripts/full_audit.py > results/full_audit_final.txt 2>&1
echo "[chain-analysis] $(date) audit written: results/full_audit_final.txt"

# 5. Combined analysis: 6-operator union across both pilots.
echo "[chain-analysis] $(date) running combined analysis"
python -u scripts/analyze_combined.py > results/combined_analysis_final.txt 2>&1
echo "[chain-analysis] $(date) combined analysis written: results/combined_analysis_final.txt"

# 6. Rigorous hypothesis check on the full clean dataset.
echo "[chain-analysis] $(date) running hypothesis_check_v2"
python -u scripts/hypothesis_check_v2.py > results/hypothesis_check_final.txt 2>&1
echo "[chain-analysis] $(date) hypothesis_check_v2 written: results/hypothesis_check_final.txt"

echo "[chain-analysis] $(date) ALL DONE."
echo "  results/full_audit_final.txt"
echo "  results/combined_analysis_final.txt"
echo "  results/hypothesis_check_final.txt"
