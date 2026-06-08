#!/bin/bash
# Final chain (resume run, 2026-06-01):
#  1. Wait for main pilot exit.
#  2. Run citation_relocation pilot synchronously.
#  3. Run combined analysis + rigorous hypothesis check on the union dataset.
#  4. NO auto-commit/push - leave that to the human for review.

set -u
cd "$(dirname "$0")/.."

PILOT_PID="${1:-$(cat /tmp/main_pilot_pid 2>/dev/null || true)}"
if [ -z "${PILOT_PID:-}" ]; then
    echo "[chain] no main pilot PID given; aborting"
    exit 1
fi

echo "[chain] $(date) waiting for main pilot PID $PILOT_PID"
while kill -0 "$PILOT_PID" 2>/dev/null; do
    sleep 60
done
echo "[chain] $(date) main pilot exited"

source .venv/bin/activate
export $(grep -v '^#' .env | xargs)

# 1. Citation_relocation pilot (synchronous - fail or finish, same watcher).
echo "[chain] $(date) launching citation_relocation pilot"
python -u scripts/run_citation_relocation_pilot.py \
    > logs/citation_relocation.log 2>&1
CR_EXIT=$?
echo "[chain] $(date) citation_relocation exited with code $CR_EXIT"

# 2. Combined analysis (folds preview_pilot + citation_relocation results).
echo "[chain] $(date) running combined analysis"
python -u scripts/analyze_combined.py \
    > results/combined_analysis.txt 2>&1
echo "[chain] $(date) combined analysis written"

# 3. Rigorous hypothesis check on the union dataset.
echo "[chain] $(date) running hypothesis_check_v2"
python -u scripts/hypothesis_check_v2.py \
    > results/hypothesis_check_final.txt 2>&1
echo "[chain] $(date) hypothesis_check_v2 written"

echo "[chain] $(date) DONE. Review results/combined_analysis.txt and results/hypothesis_check_final.txt"
