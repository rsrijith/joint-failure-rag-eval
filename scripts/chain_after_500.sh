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

# 3. Run citation_relocation pilot SYNCHRONOUSLY. This means the chain
#    process only exits after both pilots are done, so a single watcher
#    on the chain PID suffices.
echo "[chain] $(date) — running citation_relocation pilot (synchronous)"
python -u scripts/run_citation_relocation_pilot.py \
    > /tmp/pilot_citation_relocation.log 2>&1
CR_EXIT=$?
echo "[chain] $(date) — citation_relocation pilot exited with code $CR_EXIT"

# 4. Commit + push the citation_relocation results regardless of exit code,
#    so any partial progress is captured.
git add results/citation_relocation_pilot/ data/cache/expertqa_cited.jsonl 2>/dev/null
git commit -m "citation_relocation pilot complete (6th operator, ExpertQA only)" 2>/dev/null
git push 2>/dev/null
echo "[chain] $(date) — citation_relocation results committed"
echo "[chain] $(date) — DONE. Combined 6-op analysis can now run."
