#!/bin/bash
# scripts/solve_oracle_tier.sh -- run MadNLP over the external-Fortran tier.
#
#   bash scripts/solve_oracle_tier.sh            # writes runs/oracle_tier_madnlp.tsv
#   OUT_TSV=/tmp/x.tsv bash scripts/solve_oracle_tier.sh
#
# Needs MadNLP in the active environment. Resumes: problems already in the
# TSV are skipped.
#
# Smallest problems first, so the sweep yields information early, and a hard
# per-problem process timeout so a wedged Hessian costs 4 minutes, not the run.
J=$HOME/.juliaup/bin/julia
OUT=${OUT_TSV:-$PWD/runs/oracle_tier_madnlp.tsv}
[ -f "$OUT" ] || echo -e "problem\tnvar\tncon\tstatus\tobjective\titer\tbuild_s\tsolve_s" > "$OUT"
LIST=$(mktemp)
"$J" --project="$(dirname "$0")/.." -e '
using ExaCUTEst
d = classification()
sz = [(haskey(d, p) ? d[p]["variables"]["number"] + d[p]["constraints"]["number"] : typemax(Int), p)
      for p in oracle_tier_problems()]
for (_, p) in sort(sz); println(p); end' > "$LIST" || { echo "could not list problems"; exit 1; }
[ -s "$LIST" ] || { echo "empty problem list; refusing to report an empty sweep as success"; exit 1; }
for name in $(cat "$LIST"); do
    grep -q "^${name}	" "$OUT" && continue
    timeout --kill-after=20 260 "$J" "$(dirname "$0")/solve_oracle_tier.jl" "$name" \
        || echo -e "${name}\t\t\tTIMEOUT_260s\t(killed: single evaluation exceeded the budget)\t\t\t" >> "$OUT"
done
rm -f "$LIST"
echo "driver done"
