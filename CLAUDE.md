# ExaCUTEst.jl — developer guide

A thin wrapper: it locates SIF problem files, hands them to **ExaModelsSIF**
for decoding, and presents the result through a CUTEst.jl-shaped API. There is
no per-problem source here and there should never be again — the decoder
covers the whole collection, so a hand-written model would only be a second
thing to keep correct (that is what 0.1 was, and why it was removed).

```
src/
  ExaCUTEst.jl        # module: deps, the external-Fortran name list, exports
  native.jl           # NATIVE_MODELS registry; native/ holds the models
  collection.jl       # set_mastsif / list_sif_problems / sif_problem_path
  classification.jl   # select_sif_problems + classf.json loading
  classf.json         # generated; do not hand-edit
  model.jl            # ExaCUTEstModel, -param parsing, name/linearity accessors
  oracle.jl           # external-Fortran tier (CUTEst-backed ExaModels oracles)
scripts/
  build_classification.jl   # regenerates src/classf.json
test/runtests.jl      # API contract + agreement with CUTEst.jl on a sample
```

## Roadmap (Sungho, 2026-10-16) -- the plan of record

Verbatim: "verify the callbacks for the models with exernal fortran code ->
port all the fortran code to julia (perhaps in a gpu compatible way) -> verify
the pure julia version of 32 instances (callback only) -> work on gpu
compatibility, we want to be able to create a model with scalar indexing
disabled, and callbacks are verified against cpu impl (over all 15k
instances) -> the full verification again (on cpu, gpu, differenet
precisions) to make sure we have correct impl -> done!"

1. **Verify the oracle callbacks. DONE 2026-10-16: 32/32 ASSEMBLY_OK, no
   mismatches** (`test/oracle_assembly.jl`,
   `runs/oracle_assembly_2026-10-16.tsv`). Verifies our ASSEMBLY of the
   Fortran, not the Fortran: sparsity patterns, obj_weight/multiplier
   conventions, the Lagrangian Hessian split between the scalar and vector
   oracles. Comparing against CUTEst.jl cannot confirm the Fortran (both call
   it) but the wiring is ours and can be wrong; it isn't.
   Hessians are compared as Hessian-VECTOR products (Sungho: "for those just
   use hessian vectors") with three random directions at each of two points,
   plus the full matrix wherever it is cheap to materialize. That is what made
   BA-L52 checkable at all -- 192627 variables, where the full matrix would
   mean ~192k Fortran calls. Caveat: three directions is a sample, so a wrong
   row orthogonal to all six probes would slip through.
2. **Port all 32 external-Fortran problems to Julia.** THE SHAPE MATTERS, and
   getting it wrong cost a day: the port is a **generic Julia function**
   interfaced to ExaModels as an **external oracle**, NOT an ExaModels
   expression (Sungho 2026-10-16: "it should be a generic julia function" ...
   "this generic julia function can be interfaced with ExaModels as an
   external oracle"). Building the BA reprojection as a symbolic expression
   inside a SIF element did not finish compiling in an hour on a
   57-VARIABLE problem, because ExaModels nests a type parameter per
   operation and the chain is ~30 deep; the same computation as a plain
   function builds the model in 0.3-7 s. A generic function also evaluates in
   Float64/Float32/BigFloat with no extra code (step 5 for free) and, being
   allocation- and branch-free, is callable from a GPU kernel (step 4).
   Derivatives come from AD over the one function, so the Fortran's
   hand-coded G/H arrays are never ported -- the error class that made the
   legacy HS67 port wrong in the 4th digit.
   **BA-L* family DONE 2026-10-16: 12/12, all callbacks match the Fortran**
   (`src/native/BA.jl`, `test/ba_verify.jl`,
   `runs/ba_verify_2026-10-16.tsv`). Oracle tier 32 -> 20.
3. **Verify the pure-Julia versions against the Fortran callbacks** -- this
   one IS independent, unlike step 1, because the two implementations are
   genuinely different. It is what caught HS67. Done for the 12 BA-L*
   problems; 20 to go.
4. **GPU compatibility.** Acceptance criterion, stated concretely: a model
   must be constructible with **scalar indexing disabled**
   (`CUDA.allowscalar(false)`), and its callbacks verified against the CPU
   implementation **over the whole collection**, not a sample. ("15k" in the
   dictation is 1.5k, i.e. the 1548 problems.)
5. **Full verification again**: CPU, GPU, and multiple precisions.

Until step 2 lands, GPU verification stays out of scope (see below) and
solver testing is explicitly NOT wanted: "solve is not a too much concern
now; don't run solver testing" (2026-10-16). Partial solver results from the
sweep that was stopped are in runs/solve_results.s*.tsv -- 1050 of 1516
problems, kept for reference, not a deliverable.

## Readiness priorities (Sungho, 2026-10-16)

GPU verification across the collection is **deliberately not a goal at this
stage**: "I don't think it's needed at this stage; it's gonna be easy once we
complete the cpu impl". 15 problems are GPU-verified (13 in
ExaModelsSIF's test/gpusmoke.jl plus the two native models) and that is
enough of a smoke test. Do not spend compute on a full GPU build sweep, and
do not list it as an outstanding gap.

Solver testing is NOT wanted at this stage either (2026-10-16). The
oracle tier's Hessian cost -- an explicit `hess!` on ExaModels'
`ScalarNonlinearOracle`, measured as taking RAYBENDS' Hessian from 63.6 s to
~32 ms -- mattered only for solvability, so it is parked. Porting those
problems to Julia (roadmap step 2) removes the oracle Hessian problem
entirely rather than optimizing it, which is the better reason to prefer the
port.

## Two traps that cost real time on 2026-10-16

- **Any script that builds CUTEst models must hold the pidlock**
  (`mkpidlock("/tmp/cutest-decode.pid"; wait = true)`). CUTEst decodes and
  compiles every problem into ONE shared directory, so concurrent builds
  collide and report `Unable to compile a shared library for the problem` --
  which reads exactly like a broken problem and is not. Step 1 reported that
  for BA-L21LS, BA-L49, BA-L49LS and BA-L52; under the lock all four are
  ASSEMBLY_OK.
- **BA-L52 and BA-L52LS take ~1035 s to COMPILE** (the others in the family
  take 6-8 s). Any sweep with a per-problem process timeout under ~20 minutes
  records them as timeouts, which is what made them look unsolvable in the
  solver board. Budget for the build separately from the solve.
  Worth stating as a correction: a "BA-L52 cannot be built, reproduced in
  isolation" claim was recorded here earlier and was WRONG -- that probe ran
  while step 1 was concurrently using the shared compile directory, i.e. it
  failed to control for the very confound named in the bullet above.

## The native tier (src/native/)

A hand-written model lives here only if BOTH hold: the SIF uses external
Fortran (so the decoder refuses it), AND the Fortran computes something Julia
expresses exactly. Today that is `HS68` and `HS69`, whose Fortran is just the
normal CDF. They are preferred over the oracle tier, so they accept a
`backend`.

Adding one without validating it against a `CUTEstModel` is worse than
leaving it on the oracle tier, because a hand-written model is wrong
*silently*. The legacy 0.1 port offered three candidates and one failed:
`HS67` swaps the Fortran's fixed-point iteration for closed-form roots and is
off in the fourth digit. It is not a rounding artifact -- the Fortran's
derivatives are those of the finite iteration, not of the exact fixed point.
Do not resurrect it without re-deriving the iteration itself.

Never put a DECODABLE problem here. The decoder is validated over the whole
collection; a hand-written duplicate would just be a second thing to keep
correct, which is exactly what 0.2 removed.

## Division of labour

- **ExaModelsSIF** owns SIF semantics and is deliberately CUTEst-free: no
  dependency on CUTEst.jl, no Fortran toolchain, no package extension. It
  decodes, or raises `ExternalFortranError`.
- **ExaCUTEst** owns everything CUTEst-facing: the collection artifacts, the
  CUTEst.jl-compatible API, and the oracle tier for problems ExaModelsSIF
  refuses. CUTEst.jl is a plain dependency here (Sungho, 2026-10-14: the
  oracle tier "is not extension, it should be a separate repo").

A decoder fix belongs upstream in ExaModelsSIF. Only API shape, selection and
the oracle tier belong here.

## Things that will bite you

- **`nlp.meta.nlin` is 0 for every model.** ExaModels does not classify row
  linearity. `row_linearity(name)` reads it from the decode instead; the
  classification builder uses that, not `meta`. Verified on HS14 (1 linear,
  1 nonlinear) and HS35.
- **Ordering differs from CUTEst's.** CUTEst reorders variables and rows; a
  decoded model is in SIF declaration order. Any comparison against CUTEst.jl
  must permute by SIF name (`sif_variable_names`, `sif_constraint_names`);
  comparing positionally will look like a decoder bug and is not one.
- **One live Fortran model per problem name.** `cutest_oracle_model` caches
  its `CUTEstModel` in `_ORACLE_REFS`, because the oracle closures keep it
  open for the model's whole life and a second construction of the same
  problem otherwise dies with `CUTEstException(1, "memory allocation
  error")`. Measured: a repeat failed, ten different problems in a row were
  fine, and an earlier model kept evaluating correctly after a later one was
  built -- so it is specifically a repeat, not cross-contamination. The same
  omission (not finalizing) is what cost `scripts/build_classification.jl`
  two entries. `clear_oracle_cache()` releases the handles.
- **The oracle tier's objective Hessian costs one Fortran `hprod` per Hessian
  column, and that is what makes most of these problems unsolvable.**
  `ScalarNonlinearOracle` accepts `hvp!`, not an explicit sparse Hessian, so
  ExaModels reconstructs the objective block column by column (ExaModels
  `src/oracle.jl`, the `scalar_oracles` loop in `hess_coord!`). Measured
  2026-10-16, one Hessian evaluation, oracle versus CUTEst.jl's single call:

  | problem | hess cols | CUTEst | oracle | hprod x cols |
  |---|---|---|---|---|
  | PARKCH | 15 | 425 ms | 6.34 s | 9.25 s |
  | BLEACHNG | 17 | 504 ms | 8.55 s | 8.53 s |
  | RAYBENDS | 2050 | 31.6 ms | 63.6 s | 62.0 s |

  The oracle cost tracks `hprod x ncols`, which is what identifies the
  mechanism rather than merely correlating with it. RAYBENDS' single Hessian
  exceeds a 60 s solver budget, so it reports zero iterations; an explicit
  `hess!` would take it from 63.6 s to ~32 ms. Note PARKCH and BLEACHNG are
  ALSO intrinsically slow (one hprod is ~0.5 s), so both factors are real --
  do not attribute everything to the reconstruction. Constraint Hessians go
  through `hess!` in one call and are unaffected. The fix is upstream, a
  `hess!` on `ScalarNonlinearOracle`, not a workaround here.
- **Solver results for the oracle tier are archived** in
  `runs/oracle_tier_madnlp_2026-10-16.tsv`: MadNLP solves 15 of the 34, four
  exhaust a 60 s budget, 14 are killed inside a single evaluation at 260 s,
  and VANDANIUMS reports infeasible. Re-run with
  `scripts/solve_oracle_tier.sh` -- one process per problem with a hard
  timeout, because a first attempt that ran them in one process wedged on
  BA-L16LS and produced nothing in two hours.
- **The oracle tier cannot take a `backend`.** Opaque Fortran callbacks have
  no expression graph to vectorize. The constructor refuses the combination
  rather than silently building a CPU model.
- **Validating the oracle tier against CUTEst.jl is circular** — both sides
  call the same compiled routine. The census records those 34 as
  `ORACLE_TIER`, not `PASS`; the claim is that they build and evaluate, not
  that their derivatives are independently confirmed.
- **Our exports collide with CUTEst.jl's** (`list_sif_problems`,
  `select_sif_problems`, `set_mastsif`, `generator_sif_problems`). That is
  intentional — same names are what make porting a rename — but with both
  packages loaded Julia binds neither, and the error reads
  `UndefVarError: list_sif_problems not defined`, which looks like a missing
  export and is not one. The test suite imports explicitly for this reason.
- **Never hardcode an artifact's inner directory name.** These artifacts wrap
  their files in a release-named directory that changes between releases.
  Copying `optrove-maros-meszaros-9adfb5707b1e` from an older CUTEst.jl gave a
  path that does not exist, and `mastsif()` then fell back to the default
  collection, so `set_mastsif("maros-meszaros")` silently served the 1548
  NLP problems — a wrong answer whose only symptom was a plausible count.
  `_sif_dir` discovers the directory and refuses on ambiguity; `mastsif()`
  errors on a set-but-invalid `MASTSIF` rather than falling back.
- **Two collections exist and differ.** The active artifact is CUTEst.jl
  v1.4.0's `sifcollection` (1548 problems); v1.3.8's had 1540, and some shared
  problems differ in content (EQC, READING4). Pin deliberately when comparing
  results across time.

## Regenerating the classification table

Two guards exist because both failures already happened once:

- The script **refuses to write a partial table.** An earlier run wrote 1537
  of 1539 entries and reported the shortfall only as `(2 skipped)` in a
  trailing line, while the field-by-field comparison got *better* (the two
  lost problems were the only two that disagreed). A missing entry reads as
  "that problem is not in the collection", which is worse than a wrong field.
- It **cross-checks decode-derived metadata against fully built models** on a
  sample, all 11 fields, and errors if fewer than 10 problems were checked.
  That is the only thing that would catch `conbounds`/`varclasses` drifting
  from builder.jl.

The two problems it lost were `BA-L21` and `BA-L21LS`, and they build fine in
isolation: the cause was this script not finalizing its `CUTEstModel`s, so
the Fortran decoder's global state tripped up *later* constructions. If
oracle-tier entries start disappearing again, suspect finalization before
suspecting the problems.


```
julia --project=. scripts/build_classification.jl    # ~1548 decodes
```

Classification letters come from `CLASSF.DB` inside the collection artifact;
sizes and bound/row-type counts come from decoding each problem, because
`CLASSF.DB` records `V` wherever a size is a `$-PARAMETER`.

## Testing

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

Needs gfortran, since the agreement layer builds real `CUTEstModel`s. The
full-collection validation is not here — it lives in ExaModelsSIF
(`test/valsweep.jl`, 1548/1548).
