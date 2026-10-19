#!/bin/bash
# Supervise one worker: run worker.jl until its task source is empty. When the
# process dies (SIGALRM timeout, OOM kill, crash) record the problem and phase
# it was on in results/deaths.tsv, mark it done, and start a fresh process,
# which resumes after it.
#   CV_RUN=<run dir> run_worker.sh cpu cpu-<host>-07
#   CV_RUN=<run dir> run_worker.sh gpu gpu0 shards/shard_gpu0.txt <device ordinal>
set -u
F=${CV_RUN:?set CV_RUN to the run directory}
HERE=$(cd "$(dirname "$0")" && pwd)
JULIA=${JULIA:-julia}
export CV_RUN CV_MODE=$1 CV_WORKER=$2 CV_SHARD=${3:-}
[ -n "${4:-}" ] && export CUDA_VISIBLE_DEVICES=$4
export JULIA_NUM_THREADS=1
log=$F/logs/$CV_WORKER.log
blank=0
while :; do
    : > "$F/state/inflight.$CV_WORKER"
    "$JULIA" --startup-file=no --project="$HERE/../gpu" "$HERE/worker.jl" >> "$log" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    IFS=$'\t' read -r prob phase t0 < "$F/state/inflight.$CV_WORKER"
    if [ -z "$prob" ]; then
        blank=$((blank + 1))
        echo "worker exited $rc with nothing in flight ($blank)" >> "$log"
        [ $blank -ge 3 ] && break
        continue
    fi
    case $rc in
        142) st=TIMEOUT ;;
        137) st=KILLED_137 ;;
        *)   st=CRASH_$rc ;;
    esac
    now=$(date +%s)
    flock "$F/results/deaths.tsv.lock" -c "printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' '$prob' '$CV_MODE' '$st' '$phase' $((now - t0)) '$(hostname)' '$CV_WORKER' >> '$F/results/deaths.tsv'"
    [ "$CV_MODE" = gpu ] && echo "$prob" >> "$F/state/done.$CV_WORKER"
done
echo "worker $CV_WORKER finished $(date)" >> "$log"
