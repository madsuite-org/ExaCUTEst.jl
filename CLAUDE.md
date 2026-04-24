# ExaCUTEst.jl — Developer Guide

ExaCUTEst.jl converts CUTEst SIF benchmark problems into [ExaModels](https://github.com/exanauts/ExaModels.jl) format, enabling GPU-parallel automatic differentiation on the standard CUTEst test set. It is structured similarly to COPSBenchmark.jl and LuksanVlcekBenchmark.jl.

## Repository layout

```
src/
  ExaCUTEst.jl    # module: include + export all problems
  ROSENBR.jl      # each file = one SIF problem
  HS5.jl
  TAME.jl
  ALSOTAME.jl
  DEGDIAG.jl
test/
  runtests.jl     # loops over MODELS, compares every callback against CUTEst.jl
.github/
  workflows/
    test.yml      # CI: cpu (ubuntu-latest) + gpu (self-hosted cuda)
```

## How to add a new problem

1. Read the `.SIF` file in `~/git/cutest-sif/<NAME>.SIF` and translate using the SIF semantics below.
2. Create `src/<NAME>.jl` with a function `<lowercase>_model(; T = Float64, backend = nothing, kwargs...)`. Use only the `@add_var`, `@add_obj`, `@add_con` macro forms.
3. Add it to `src/ExaCUTEst.jl`: `include` + `export`.
4. Add an entry to the `MODELS` constant in `test/runtests.jl`:
   ```julia
   ("<NAME>", b -> <lowercase>_model(; backend = b)),
   ```
   The test loop calls `test_cutest_callbacks` on every entry automatically.

See `src/ROSENBR.jl` as the simplest example, `src/TAME.jl` for equality constraints, `src/ALSOTAME.jl` for nonlinear constraints, and `src/DEGDIAG.jl` for a large scalable problem with DO/OD loops.

## Always use macro syntax

Always use the ExaModels macro forms — these are the first-class API:
```julia
ExaModels.@add_var(c, x, N; lvar = ..., uvar = ..., start = ...)
ExaModels.@add_obj(c, expr)                  # scalar term
ExaModels.@add_obj(c, expr for i in iter)    # sum over loop
ExaModels.@add_con(c, expr)                  # scalar constraint
ExaModels.@add_con(c, expr for i in iter)    # vectorised constraints
```
Never use the functional forms (`add_var`, `add_obj`, `add_con`).

## CI structure

| Job | Runner | Backends tested |
|-----|--------|-----------------|
| `cpu` | `ubuntu-latest` (GitHub-hosted) | `nothing`, `CPU()` |
| `gpu` | `self-hosted, cuda` | `nothing`, `CPU()`, `CUDABackend()` |

The test runner uses `CUDA.functional()` to auto-detect GPU availability, so the same `test/runtests.jl` works for both jobs.

## Test pattern

Every problem in `MODELS` is tested by `test_cutest_callbacks(name, build_model)` which:
1. Loads `CUTEstModel(name)` as ground truth.
2. For each backend, builds the ExaModel, wraps it with `ExaModels.WrapperNLPModel` (which moves GPU outputs back to CPU), then checks:
   - `nvar`, `ncon`, `nnzj`, `nnzh` match CUTEst
   - `obj`, `grad`, `cons` at the starting point match CUTEst (atol=1e-10)
   - `jac_structure` + `jac_coord`: assembled sparse Jacobian matches CUTEst (norm < 1e-10)
   - `hess_structure` + `hess_coord`: assembled sparse Hessian matches CUTEst (norm < 1e-10)

## SIF format — key semantics

### Group structure

Every SIF group defines either an objective term (N type) or a constraint (E/L/G type). The group value is:

```
v_g = linear_part(x) + element_contributions(x) - constant_rhs
```

If a `'SCALE'` σ is given, the **group function** receives `v_g / σ` and the full contribution is `σ · φ(v_g / σ)`:

For GROUP TYPE `L2` (φ(u) = u²): `contribution = σ · (v/σ)² = v²/σ`.

Examples:
- ROSENBR G1: σ=0.01, v = x2 − x1²  →  contribution = (x2−x1²)² / 0.01 = **100(x2−x1²)²** ✓
- HS3 G2: σ=100000, v = x2−x1  →  contribution = (x2−x1)² / 100000 = **10⁻⁵(x2−x1)²** ✓

### Variable bounds default

When a SIF problem has no BOUNDS section (or no entry for a variable), the default is **x ≥ 0** (not free). A `FR` bound makes a variable free. Set `lvar` accordingly; see TAME for an example.

### Constraint convention (matching CUTEst)

CUTEst.jl reports constraint values as `g(x) − b = 0` (residual form). Encode ExaModels constraints the same way so the values match:
```julia
ExaModels.@add_con(c, expr - rhs)   # lcon=0, ucon=0 by default → equality
```
For inequality groups (L type ≤ 0, G type ≥ 0), set `lcon`/`ucon` appropriately.

### GROUP TYPE functions

| SIF type | φ(v) |
|----------|------|
| L2 / SQUARE | v² |
| SINE | sin(v) |
| COS | cos(v) |
| EXP / EXPN | exp(v) |

Without a GROUP TYPE the function is the identity (φ(v) = v), i.e., a pure linear/element sum.

### ELEMENT TYPE

Element types define nonlinear functions of one or more variables. The element value is added (with a coefficient) to the group's nonlinear part. Common elements:
- `SQ(V1)` = V1²  (coefficient in GROUP USES is the multiplier)

### QUADRATIC section

Entries `X X(I) X(J) coeff` add **0.5 × coeff × x_i × x_j** to the objective (Hessian convention). A diagonal entry `X(I) X(I) 1.0` → **0.5 × x_i²**, not **x_i²**. See DEGDIAG.

### DO/OD loops and parameter arithmetic

Scalable SIF problems use DO/OD loops with integer parameters (IE) and real arithmetic (RE, RI, RA, R/). Replicate these in Julia with `[expr for i in range]`. See DEGDIAG for how to compute `lvar = T[i/(N+1) for i in 0:N]`.

Note: DEGDIAG lvar comparison with CUTEst requires `rtol=1e-10` rather than `atol` because Fortran integer-to-float conversion in sifdecoder introduces tiny rounding differences vs Julia.

## SIF feature conversion difficulty

| Feature | Difficulty | Examples |
|---------|-----------|---------|
| Fixed-size, simple group types | Easy | ROSENBR, HS5, TAME, ALSOTAME |
| Scalable DO/OD loops | Medium | DEGDIAG, GOFFIN |
| Element types with multiple variables | Medium | HS1, many HS problems |
| Parameter arithmetic (RI/RA/R/) | Medium | DEGDIAG bounds |
| TEMPORARIES in elements/groups | Hard | ALSOTAME (sin/exp) |
| QUADRATIC section | Medium | DEGDIAG |
| RANGES for inequality constraints | Easy | most HS problems |
