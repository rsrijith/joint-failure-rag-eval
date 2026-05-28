#!/bin/bash
# Chain: wait for current 75-seed 8-judge pilot to exit, run analysis +
# commit, then auto-launch the full 500-seed pilot.
#
# Run from repo root with the venv activated:
#   bash scripts/chain_75_then_500.sh > /tmp/chain.log 2>&1 &
#   disown

set -u
cd "$(dirname "$0")/.."

# 1. Find the currently-running pilot PID
PILOT_PID=$(pgrep -f "Resources/Python.*run_preview_pilot" | head -1 || true)
if [ -z "$PILOT_PID" ]; then
    echo "[chain] no pilot running; skipping wait"
else
    echo "[chain] waiting for pilot PID $PILOT_PID to exit..."
    while kill -0 "$PILOT_PID" 2>/dev/null; do
        sleep 60
    done
    echo "[chain] $(date) — pilot exited"
fi

# 2. Run analysis on the 75-seed dataset
echo "[chain] $(date) — running analysis on current dataset"
python scripts/analyze_preview_pilot.py > results/preview_pilot/analysis_8judges_75seeds.txt 2>&1
echo "[chain] analysis saved to results/preview_pilot/analysis_8judges_75seeds.txt"

# 3. Commit and push the 75-seed 8-judge results
git add results/preview_pilot/ 2>/dev/null
git commit -m "8-judge x 75-seed pilot complete (preview)" 2>/dev/null
git push 2>/dev/null
echo "[chain] $(date) — committed preview results"

# 4. Launch the full 500-seed pilot
echo "[chain] $(date) — launching full 500-seed pilot"
nohup python -u scripts/run_preview_pilot.py > /tmp/pilot_500.log 2>&1 &
FULL_PID=$!
disown
echo "[chain] $(date) — full pilot pid=$FULL_PID, log /tmp/pilot_500.log"
echo "$FULL_PID" > /tmp/full_pilot_pid
