#!/bin/bash
# Run on a GPU host after starting every host's launch_host.sh:
#   CV_RUN=<run dir> setsid nohup finish.sh HOST... &
# Blocks on each listed host's run lock (no polling), then retries each problem
# that hit OOM_GPU once, on this host's GPU with the most free memory at that
# moment (rows to results/gpu_retry.tsv), writes the analysis to
# results/report.txt, and touches state/FINISHED.
set -u
F=${CV_RUN:?set CV_RUN to the run directory}
HERE=$(cd "$(dirname "$0")" && pwd)
JULIA=${JULIA:-julia}
cd "$F"
exec 8> state/finish.lock   # held until this script exits
flock 8
for h in "$@"; do
    flock "state/running.$h.lock" true
done
echo "workers end $(date +%s) $(date)" >> state/times.txt
awk -F'\t' '$7=="OOM_GPU"{print $1}' results/gpu.tsv 2>/dev/null | sort -u > shards/retry_oom.txt
if [ -s shards/retry_oom.txt ]; then
    dev=$("$JULIA" --startup-file=no --project="$HERE/../gpu" -e '
        using CUDA
        free = map(collect(CUDA.devices())) do d
            CUDA.device!(d); (CUDA.free_memory(), CUDA.deviceid(d))
        end
        print(last(maximum(free)))' 2> state/gpumem_retry.err)
    echo "retry start $(date +%s) device $dev $(date)" >> state/times.txt
    CV_TSV=gpu_retry.tsv "$HERE/run_worker.sh" gpu gpur shards/retry_oom.txt "$dev"
    echo "retry end $(date +%s) $(date)" >> state/times.txt
fi
python3 -I "$HERE/analyze.py" "$F" > results/report.txt 2> results/report.err
echo "analysis end $(date +%s) $(date)" >> state/times.txt
touch state/FINISHED
