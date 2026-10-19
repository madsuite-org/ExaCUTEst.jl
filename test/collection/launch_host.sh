#!/bin/bash
# Start this host's workers and wait for all of them, holding
# state/running.<host>.lock for the whole time, so that
#   flock state/running.<host>.lock true
# blocks exactly until this host has finished. Run detached on every host:
#   CV_RUN=<run dir> setsid nohup launch_host.sh NCPU GPUSPEC... &
# GPUSPEC = worker:shardfile:device, e.g. gpu0:shards/shard_gpu0.txt:0
set -u
F=${CV_RUN:?set CV_RUN to the run directory}
HERE=$(cd "$(dirname "$0")" && pwd)
H=$(hostname)
ncpu=$1; shift
mkdir -p "$F/state" "$F/results" "$F/logs"
exec 9> "$F/state/running.$H.lock"
flock 9
echo "$H start $(date +%s) $(date)" >> "$F/state/times.txt"
for spec in "$@"; do
    IFS=: read -r w shard dev <<< "$spec"
    "$HERE/run_worker.sh" gpu "$w" "$shard" "$dev" 9>&- &
done
for i in $(seq -w 1 "$ncpu"); do
    "$HERE/run_worker.sh" cpu "cpu-$H-$i" 9>&- &
done
wait
echo "$H end $(date +%s) $(date)" >> "$F/state/times.txt"
