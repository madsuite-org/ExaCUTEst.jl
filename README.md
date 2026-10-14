# ExaCUTEst.jl

The [CUTEst](https://github.com/ralna/CUTEst) test set as [ExaModels](https://github.com/exanauts/ExaModels.jl) — on the CPU or on a GPU — with an API that mirrors [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl).

```julia
using ExaCUTEst, NLPModels

nlp = ExaCUTEstModel("ROSENBR")
obj(nlp, nlp.meta.x0)                       # the NLPModels API, as usual

nlp = ExaCUTEstModel("READING4", "-param", "N=1000")   # size parameters
nlp = ExaCUTEstModel("DIXMAANA1"; backend = CUDABackend())   # on the GPU
```

Problems are read from their SIF files by [ExaModelsSIF.jl](https://github.com/madsuite-org/ExaModelsSIF.jl), a SIF front end for ExaModels, so the whole collection is available and each problem arrives as a SIMD-abstracted ExaModel with automatic differentiation on every backend ExaModels supports (CUDA, AMDGPU, oneAPI, Metal).

## Relationship to CUTEst.jl

CUTEst.jl calls the Fortran CUTEst library: it decodes each SIF file with `sifdecoder`, compiles a shared library, and evaluates through Fortran callbacks on the CPU. ExaCUTEst decodes the same SIF files into ExaModels expression graphs in Julia.

| | CUTEst.jl | ExaCUTEst.jl |
|---|---|---|
| evaluation | compiled Fortran callbacks | ExaModels SIMD kernels |
| GPU | no | yes, via `backend` |
| element type | `Float32`/`Float64`/`Float128` | any, incl. `BigFloat` |
| build step | `sifdecoder` + gfortran per problem | none |
| concurrent models | one per name (global decoder state) | unrestricted |

What ports directly: the constructor (`CUTEstModel` → `ExaCUTEstModel`), its `-param` arguments, `set_mastsif`, `list_sif_problems`, `generator_sif_problems`, `select_sif_problems` with all its keywords, and the NLPModels interface. `finalize(nlp)` still runs and is simply unnecessary.

Three real differences to know about:

- **Ordering.** CUTEst reorders variables (nonlinear first) and rows (equality and linear first); a decoded model keeps SIF declaration order. Use `sif_variable_names` and `sif_constraint_names` to map one indexing onto the other — that is exactly what the test suite does to compare the two.
- **Row linearity.** `nlp.meta.nlin` is `0` for every model, because ExaModels does not classify rows. Use `row_linearity(name)`, which reads it from the decode.
- **External Fortran.** 34 problems append hand-coded Fortran element/group functions to their SIF file (`external_fortran_problems()`). Those are not expressions, so they are evaluated through the compiled Fortran, wrapped as ExaModels oracles: CPU only, `Float64` only, and `backend` is refused for them. `external = :error` rejects them instead. Their objective Hessian is reconstructed one Hessian-column at a time (ExaModels' `ScalarNonlinearOracle` takes a Hessian-vector product, not an explicit sparse Hessian), so it costs one Fortran `hprod` per column: measured 4x CUTEst.jl's `hess_coord` at 2 columns and 24x at 57, which makes the largest of these problems (BA-L21, 34134 variables) impractical for Hessian-heavy use until `ScalarNonlinearOracle` accepts an explicit Hessian upstream. Values, gradients, constraints and Jacobians carry no such penalty.

One sharp edge: ExaCUTEst exports the same names as CUTEst.jl, which is what
makes porting a rename — but it also means that with **both** packages loaded
Julia leaves the shared names (`list_sif_problems`, `select_sif_problems`,
`set_mastsif`, `generator_sif_problems`) unbound, and calls must be qualified
as `ExaCUTEst.select_sif_problems(...)`. Only code comparing the two ever
loads both.

The library-management functions (`sifdecoder`, `build_libsif`, `clear_libsif`, `manage_libsif`) have no counterpart, because nothing is compiled and no shared library is written.

## Problem collections

The collections are Julia artifacts — the same ones CUTEst.jl v1.4.0 declares, so both packages read byte-identical problem files. `ENV["MASTSIF"]` names the active directory and is honored if set before loading.

```julia
set_mastsif()                   # "sifcollection", the CUTEst NLP set (1548 problems)
set_mastsif("maros-meszaros")   # convex QPs
set_mastsif("netlib-lp")        # LPs
```

## Selecting problems

```julia
select_sif_problems(; min_var = 10, max_var = 100, only_linear_con = true)
select_sif_problems(; objtype = :sum_of_squares)
filter(is_truly_unconstrained, select_sif_problems())
```

Selection reads `src/classf.json`, built by `scripts/build_classification.jl`: classification letters from the collection's `CLASSF.DB`, sizes and bound/row-type counts from decoding every problem. Regenerating it needs no Fortran toolchain for the 1509 decodable problems in the table; the other 30 (external Fortran, of the 34 total — 4 are absent from `CLASSF.DB`) take their metadata from the compiled routine.

That table is checked against CUTEst.jl's own Fortran-derived `classf.json` by `scripts/compare_classification.jl`: **identical on all 19 fields across all 1539 problems CUTEst.jl classifies**. Ours covers all 1548 — nine problems (the `BA-L16`/`BA-L52` pairs, `LRCOVTYPE`, the four `MNISTS*`) are absent from `CLASSF.DB` and so unselectable in CUTEst.jl, and are picked up here from their SIF files' own classification lines.

## Correctness

ExaModelsSIF's decoder is validated against the Fortran reference over the entire collection — 1548/1548, comparing values, gradients, constraints, Jacobians and Lagrangian Hessians at two points per problem. A handful of disagreements were adjudicated to reference-side defects, reported upstream as [SIFDecode#55](https://github.com/ralna/SIFDecode/issues/55) and [#56](https://github.com/ralna/SIFDecode/issues/56). This package's own suite (`test/runtests.jl`) covers the wrapper API and re-checks agreement with CUTEst.jl on a sample.

## History

Versions through 0.1 shipped one hand-written Julia file per problem, around 900 of them. Version 0.2 replaced that with the decoder: it covers every problem rather than a subset, and there is no per-problem source to keep in sync with the SIF files.

## References

- N. I. M. Gould, D. Orban, Ph. L. Toint, *CUTEst: a Constrained and Unconstrained Testing Environment with safe threads for mathematical optimization*, Computational Optimization and Applications 60(3), 545–557, 2015.
- A. R. Conn, N. I. M. Gould, Ph. L. Toint, *LANCELOT: A Fortran Package for Large-scale Nonlinear Optimization (Release A)*, Springer, 1992.
- S. Shin, F. Pacaud, M. Anitescu, *Accelerating optimal power flow with GPUs: SIMD abstraction of nonlinear programs and its application to stochastic optimal power flow*, HICSS-56, 2023.

## License

MIT; see [LICENSE](LICENSE). No CUTEst Fortran or C source is included or distributed. The SIF problem files are distributed by the CUTEst project under their own terms and are downloaded as artifacts.
