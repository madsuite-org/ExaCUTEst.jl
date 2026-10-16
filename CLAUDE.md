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
