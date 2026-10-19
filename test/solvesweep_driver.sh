#!/bin/bash
# Parallel driver for test/solvesweep.jl.
#   SWEEP_PROJECT=<env with ExaCUTEst and MadNLP> bash test/solvesweep_driver.sh [nshards]
# Each shard runs under `timeout`, so a wedged evaluation kills one pass; the
# next pass attributes that problem from its current.<shard>.txt and moves on.
cd "$(dirname "$0")/.." || exit 1
N=${1:-16}
J="${JULIA:-julia}"
PASS_TIMEOUT=${PASS_TIMEOUT:-1800}
oneshard() {
    local s=$1
    for i in $(seq 1 200); do
        SWEEP_SHARD=$s SWEEP_NSHARDS=$N timeout --kill-after=30 "$PASS_TIMEOUT" \
            "$J" --project="${SWEEP_PROJECT:-.}" test/solvesweep.jl >> "runs/solve_progress.$s.log" 2>&1
        rc=$?
        echo "driver[$s]: pass $i rc=$rc" >> "runs/solve_progress.$s.log"
        [ "$rc" -eq 42 ] && continue      # chunk boundary: keep going
        [ "$rc" -eq 0 ] && return 0       # shard finished
        [ "$rc" -eq 124 ] && continue     # pass timed out: resume past it
        sleep 3
    done
    return 1
}
mkdir -p runs
pids=()
for s in $(seq 0 $((N-1))); do oneshard "$s" & pids+=($!); done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
echo "solvesweep merged $(cat runs/solve_results.s*.tsv 2>/dev/null | wc -l) results; fail=$fail"
