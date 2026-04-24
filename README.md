# ExaCUTEst.jl

ExaCUTEst.jl provides CPU/GPU-compatible [ExaModels](https://github.com/exanauts/ExaModels.jl) implementations of benchmark problems from the [CUTEst](https://github.com/ralna/CUTEst) test set. It is designed to complement [CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl), which already gives a CPU NLPModels interface via Fortran; ExaCUTEst gives the same problems as native Julia ExaModels that can run on GPU backends (CUDA, AMDGPU, oneAPI, Metal) with full automatic differentiation support.

## Background

**CUTEst** (Constrained and Unconstrained Testing Environment with safe threads) is the standard test collection for nonlinear optimization software, originally developed by Gould, Orban, and Toint. Problems are specified in the SIF (Standard Input Format) language:

- N. I. M. Gould, D. Orban, and Ph. L. Toint, *CUTEst: a Constrained and Unconstrained Testing Environment with safe threads for mathematical optimization*, Computational Optimization and Applications 60(3), pp. 545–557, 2015. DOI: [10.1007/s10589-014-9687-3](https://doi.org/10.1007/s10589-014-9687-3)
- A. R. Conn, N. I. M. Gould, and Ph. L. Toint, *LANCELOT: A Fortran Package for Large-scale Nonlinear Optimization (Release A)*, Springer, 1992.

The CUTEst Fortran/C library is licensed under the **GNU Lesser General Public License v3.0 (LGPL-3.0)**. ExaCUTEst.jl does not copy or wrap any CUTEst source code; it independently reimplements the mathematical problem formulations described in the SIF files using Julia and ExaModels.

**CUTEst.jl** (MPL-2.0) provides Julia bindings to the Fortran CUTEst library and is used here as the ground-truth reference for correctness testing:

- D. Orban and contributors, *CUTEst.jl: Julia interface to CUTEst*, [github.com/JuliaSmoothOptimizers/CUTEst.jl](https://github.com/JuliaSmoothOptimizers/CUTEst.jl)

**HS problems** (HS1–HS112) originate from the classic test collection by Hock and Schittkowski:

- W. Hock and K. Schittkowski, *Test Examples for Nonlinear Programming Codes*, Lecture Notes in Economics and Mathematical Systems, vol. 187, Springer, 1981.

**ExaModels.jl** is the symbolic algebraic modeling framework that makes SIMD-parallel evaluation and automatic differentiation possible across CPU and GPU backends:

- S. Shin, F. Pacaud, and M. Anitescu, *Accelerating optimal power flow with GPUs: SIMD abstraction of nonlinear programs and its application to stochastic optimal power flow*, in Proc. of the 56th Hawaii International Conference on System Sciences (HICSS-56), 2023.
- ExaModels.jl repository: [github.com/exanauts/ExaModels.jl](https://github.com/exanauts/ExaModels.jl)

ExaCUTEst.jl was developed by Sungho Shin with assistance from [Claude](https://claude.ai) (Anthropic).

## License

ExaCUTEst.jl is released under the **MIT License**. See [LICENSE](LICENSE) for details.

The mathematical problem formulations implemented here derive from published academic works (cited above) and the CUTEst SIF problem collection. No CUTEst Fortran/C source code is included or distributed.

## Installation

```julia
using Pkg
Pkg.add("ExaCUTEst")
```

## Usage

Each CUTEst problem is available as a function that returns an [`ExaModel`](https://github.com/exanauts/ExaModels.jl):

```julia
using ExaCUTEst, ExaModels, NLPModels

# Build on default CPU
m = rosenbr_model()
obj(m, m.meta.x0)       # objective at starting point
grad(m, m.meta.x0)      # gradient
hess_structure(m)        # (rows, cols) of Hessian sparsity pattern
hess_coord(m, m.meta.x0, zeros(0))  # Hessian values

# Build for GPU (requires CUDA.jl)
using CUDA
m_gpu = rosenbr_model(; backend = CUDABackend())

# Wrap for uniform CPU-side callback access
mw = ExaModels.WrapperNLPModel(m_gpu)
obj(mw, m.meta.x0)
```

### Available problems

| Function | CUTEst name | Variables | Constraints | Notes |
|----------|-------------|-----------|-------------|-------|
| `rosenbr_model` | ROSENBR | 2 | 0 | Rosenbrock "banana valley" |
| `hs5_model` | HS5 | 2 | 0 | H&S #5; sin/cos objective, bounds |
| `tame_model` | TAME | 2 | 1 | Equality constraint, L2 group type |
| `alsotame_model` | ALSOTAME | 2 | 1 | Exp objective, sin constraint, bounds |
| `degdiag_model(N)` | DEGDIAG | N+1 | 0 | Diagonal QP; default N=100000 |

### SIF → ExaModels key conventions

ExaCUTEst encodes SIF semantics as follows:

- **Group scale**: for a group with scale σ and L2 type, contribution = v²/σ (not σ·v²). ROSENBR G1 has σ=0.01 → factor 100.
- **Default bounds**: variables with no BOUNDS section default to x ≥ 0 in SIF.
- **Constraint form**: expressed as `g(x) − rhs` with `lcon = ucon = 0` to match CUTEst.jl's residual convention.
- **QUADRATIC section**: `X X(I) X(I) c` adds 0.5·c·x_i² (Hessian entry convention).

## Testing

```julia
julia --project=. test/runtests.jl
```

Tests compare `obj`, `grad`, `cons`, `jac_structure`, `jac_coord`, `hess_structure`, and `hess_coord` against `CUTEstModel` at the starting point, for each registered backend.

## Contributing

To add a new problem:

1. Read the `.SIF` file from [cutest-sif](https://github.com/ralna/SIF) and interpret the group/element structure per `CLAUDE.md`.
2. Create `src/<NAME>.jl` following the pattern in `src/ROSENBR.jl`.
3. Include it in `src/ExaCUTEst.jl` and add to the exports.
4. Add a test in `test/runtests.jl`.
