#!/bin/bash
# Chain: after the restarted 500-seed pilot exits, run analysis + commit,
# then launch the citation_relocation pilot.
#
# Run from repo root with the venv activated:
#   bash scripts/chain_after_500.sh > /tmp/chain_after_500.log 2>&1 &
#   disown

set -u
cd "$(dirname "$0")/.."

PILOT_PID=$(cat /tmp/full_pilot_pid_v2 2>/dev/null || true)
if [ -z "${PILOT_PID:-}" ]; then
    echo "[chain] /tmp/full_pilot_pid_v2 missing; aborting"
    exit 1
fi

echo "[chain] $(date) — waiting for main pilot PID $PILOT_PID"
while kill -0 "$PILOT_PID" 2>/dev/null; do
    sleep 60
done
echo "[chain] $(date) — main pilot exited"

# 1. Run analysis on the 500-seed 5-operator dataset.
echo "[chain] $(date) — running 5-operator analysis"
python scripts/analyze_preview_pilot.py > results/preview_pilot/analysis_8judges_500seeds.txt 2>&1 \
    && echo "[chain] analysis written: results/preview_pilot/analysis_8judges_500seeds.txt"

# 2. Commit + push the 500-seed results.
git add results/preview_pilot/ 2>/dev/null
git commit -m "8-judge x 500-seed pilot complete (5 operators)" 2>/dev/null
git push 2>/dev/null
echo "[chain] $(date) — 500-seed results committed"

# 3. Launch the citation_relocation pilot.
echo "[chain] $(date) — launching citation_relocation pilot"
nohup python -u scripts/run_citation_relocation_pilot.py \
    > /tmp/pilot_citation_relocation.log 2>&1 &
CR_PID=$!
disown
echo "$CR_PID" > /tmp/citation_relocation_pid
echo "[chain] $(date) — citation_relocation pilot pid=$CR_PID, log /tmp/pilot_citation_relocation.log"
