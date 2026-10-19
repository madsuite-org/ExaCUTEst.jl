# Collection verification

Every problem of `list_sif_problems()`, through `ExaCUTEstModel`, in four
configurations, at `x0` and a seeded displaced point `xp`:

| configuration      | test                  | reference              |
|--------------------|-----------------------|------------------------|
| `cpu64_vs_fortran` | CPU Float64           | CUTEst's Fortran (pilot.jl / native_verify.jl rules) |
| `gpu64_vs_cpu64`   | CUDA Float64          | CPU Float64            |
| `gpu32_vs_cpu32`   | CUDA Float32          | CPU Float32            |
| `cpu32_vs_cpu64`   | CPU Float32           | CPU Float64            |

Value callbacks compared: `obj grad cons jac_coord jac_asm hess_coord hess_asm
hprod`; structures must match exactly. It runs in the `test/gpu` environment,
on several hosts at once, all sharing one run directory `CV_RUN` on a
filesystem with working `flock` across hosts (NFSv4):

```
export CV_RUN=/path/on/nfs/run1 JULIA=~/.juliaup/bin/julia
C=test/collection
$JULIA --project=test/gpu $C/prepare.jl        # population.tsv, provenance.txt, device check (GPU host)
$C/sigscan.sh 16                               # sigs.tsv (or copy a previous run's: it depends only on the decoders)
python3 -I $C/plan.py $CV_RUN 4 --prev <previous run> --big-shards 3,2
# on each host, detached; GPUSPEC = worker:shardfile:device
CV_RUN=$CV_RUN setsid nohup $C/launch_host.sh 40 gpu0:shards/shard_gpu0.txt:0 gpu1:shards/shard_gpu1.txt:1 &
CV_RUN=$CV_RUN setsid nohup $C/launch_host.sh 24 gpu2:shards/shard_gpu2.txt:0 gpu3:shards/shard_gpu3.txt:1 &
# on one GPU host: waits for the listed hosts, retries GPU OOMs, writes results/report.txt
CV_RUN=$CV_RUN setsid nohup $C/finish.sh <host1> <host2> &
```

`state/FINISHED` marks completion; `python3 -I $C/analyze.py $CV_RUN` re-derives
the report from the rows at any time, including mid-run. Two things keep the
run tractable and correct:

- **Signature-affinity sharding.** GPU time is dominated by compiling each
  distinct ExaModels block signature once per process, so `plan.py` places
  problems that share signatures in the same long-lived device worker. Within
  a shard problems run smallest first; with `--prev`, the largest and
  previously OOM or slow problems go to `--big-shards` (the devices with the
  most free memory). CPU workers on every host draw from one shared queue,
  longest first.
- **The CUTEst lock.** CUTEst decodes and compiles every problem into one build
  directory (`CUTEst.libsif_path`), and concurrent builds there corrupt each
  other. Every `CUTEstModel` construction, on every host, holds an exclusive
  flock on `CV_CUTEST_LOCK` (default `$CV_RUN/cutest.lock`). Nothing else may
  build CUTEst problems against the same depot while a run is going.

Each problem is bounded by `CV_TIMEOUT` (3600 s); a timeout, OOM kill, or crash
is recorded in `results/deaths.tsv` with the phase it hit, and the worker
restarts after it. `provenance.txt` records the commit or tree hash of every
package under test; the run measures whatever those paths hold, so leave them
untouched until it ends (a dedicated worktree is the simple way).

**Verdicts**, per problem and configuration: `PASS`; `PASS_CONDITIONED`
(passes, but some reading exceeds the ulp threshold and is attributed to
conditioning, below); `FAIL`; `FIXED_ERROR` (a failure with the fixed-error
signature, below); `ZERO_REF` (listed exception, below); `TIMEOUT`;
`RESOURCE` (GPU OOM or host OOM kill); `SKIPPED` (nothing non-trivial to
compare); `NOT_RUN`.

**Criterion.** A reading's error is `max|test - ref| / (eps(T) * S)` ulp, with
`T` the lower precision of the pair and `S` the largest finite `|ref|` that
quantity reaches over both points. A reading passes at <= 1000 ulp. Above
that, the GPU configurations look at the precision ratio

    r = ulp(gpu32_vs_cpu32) / ulp(gpu64_vs_cpu64)

for the same problem and callback, each in its own eps units. Error that comes
from conditioning scales with each precision's rounding, so `r` is O(1); an
error that is a fixed wrong value is the same in both precisions, so
`r = eps64/eps32 = 1.86e-9`. Hence:

- `r` in `[1e-5, 1e4]` -> `PASS_CONDITIONED`; outside, or `r` not formable
  (either side missing, exactly zero, or non-finite) -> `FAIL`. The absence
  of the discriminator is not evidence of conditioning.
- Whatever the ulp, `r < 1e-6` with a Float64 error >= 10 ulp ->
  `FIXED_ERROR`. The 10-ulp floor matters: a Float64 difference under one ulp
  against a near-exact Float32 comparison (MISRA1CLS and LIN `hprod`, 0.7
  ulp) gives an arbitrarily small `r` that means nothing.

The bounds come from the collection on the GV100s: readings above 1000 ulp
have `r` between 1.2e-4 (DALLASM `hprod`) and ~2e2 (CHARDIS12 `obj`, Float32
configuration); among readings with a Float64 error of 10 ulp or more,
nothing lies between 1.86e-9 and 1.2e-4. The lower bound sits a decade under
DALLASM. A device Hessian whose diagonal is off by a constant amount (the
VANDERM1-4 defect in ExaModels' device Hessian) reads `r = 1.86e-9` and comes
out `FIXED_ERROR`: that is the case the window must never pass.

`cpu32_vs_cpu64` stays on the plain 1000-ulp threshold; its ratio against
`gpu64_vs_cpu64` is printed with each failure but does not decide. That
comparison also rounds `x` itself to Float32, an input perturbation the
Float64 GPU comparison never sees, so its ratios over the failing readings run
continuously from ~0.3 to ~4e5 with no gap to put a bound in, and where
`gpu64_vs_cpu64` is exactly zero no ratio exists at all. Its failures are
Float32 conditioning or Float32 overflow (non-finite entries where Float64 is
finite); they are listed in the report, not excused.

`ZERO_REF_EXCEPTIONS` in `analyze.py` names the readings whose reference is
identically zero at a point, so that no relative error exists there (TAX13322,
TAXR13322, TAX53322, TAXR53322, GOULDQP2, GOULDQP3: `cons` at an `x0` that
satisfies the constraints exactly), each with its reason; the cross-point ulp
test still applies to them. An unlisted `ZERO_REF` is a failure.
