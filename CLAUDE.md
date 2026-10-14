# ExaCUTEst.jl — developer guide

A thin wrapper: it locates SIF problem files, hands them to **ExaModelsSIF**
for decoding, and presents the result through a CUTEst.jl-shaped API. There is
no per-problem source here and there should never be again — the decoder
covers the whole collection, so a hand-written model would only be a second
thing to keep correct (that is what 0.1 was, and why it was removed).

```
src/
  ExaCUTEst.jl        # module: deps, the external-Fortran name list, exports
  collection.jl       # set_mastsif / list_sif_problems / sif_problem_path
  classification.jl   # select_sif_problems + classf.json loading
  classf.json         # generated; do not hand-edit
  model.jl            # ExaCUTEstModel, -param parsing, name/linearity accessors
  oracle.jl           # external-Fortran tier (CUTEst-backed ExaModels oracles)
scripts/
  build_classification.jl   # regenerates src/classf.json
test/runtests.jl      # API contract + agreement with CUTEst.jl on a sample
```

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
  column.** `ScalarNonlinearOracle` accepts `hvp!`, not an explicit sparse
  Hessian, so ExaModels reconstructs the objective block column by column
  (`src/oracle.jl` in ExaModels, the `scalar_oracles` loop in `hess_coord!`).
  Measured against CUTEst.jl's `hess_coord`: x4 at 2 columns (AKIVA), x7 at 6
  (HS92), x24 at 57 (BA-L1LS). Constraint Hessians go through `hess!` in one
  call and are unaffected. The fix is upstream -- `ScalarNonlinearOracle`
  would need a `hess!` -- not a workaround here.
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
