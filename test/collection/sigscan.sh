#!/bin/bash
# Scan the population's kernel signatures into CV_RUN/sigs.tsv, NPROC slices in
# parallel, re-entering sigscan.jl per chunk under a timeout so one wedged
# decode cannot stall a slice.
#   CV_RUN=<run dir> sigscan.sh [NPROC]
# The table depends only on the decoders, so a previous run's sigs.tsv can be
# copied in instead while they are unchanged.
set -u
F=${CV_RUN:?set CV_RUN to the run directory}
HERE=$(cd "$(dirname "$0")" && pwd)
JULIA=${JULIA:-julia}
n=${1:-8}
mkdir -p "$F/sigscan"
tail -n +2 "$F/population.tsv" | cut -f1 | awk -v n="$n" -v d="$F/sigscan" '{print > (d "/list." (NR % n))}'
for list in "$F"/sigscan/list.*; do
    k=${list##*.}
    (
        export SIGSCAN_LIST=$list SIGSCAN_OUT=$F/sigscan/sigs.$k.tsv
        for pass in $(seq 1 500); do
            timeout 1800 "$JULIA" --startup-file=no --project="$HERE/../gpu" "$HERE/sigscan.jl" \
                >> "$F/sigscan/log.$k" 2>&1
            rc=$?
            [ $rc -eq 42 ] && continue
            [ $rc -eq 0 ] && break
            echo "pass $pass exited $rc" >> "$F/sigscan/log.$k"
        done
    ) &
done
wait
cat "$F"/sigscan/sigs.*.tsv > "$F/sigs.tsv"
echo "sigs.tsv: $(cut -f1 "$F/sigs.tsv" | sort -u | wc -l) problems"
