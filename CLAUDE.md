# ExaCUTEst.jl

ExaCUTEst.jl converts CUTEst SIF benchmark problems into [ExaModels](https://github.com/exanauts/ExaModels.jl) format, enabling GPU-parallel automatic differentiation on the standard CUTEst test set.

## Project goal

Same idea as COPSBenchmark.jl and LuksanVlcekBenchmark.jl, but for the ~1151 CUTEst SIF problems. CUTEst.jl already provides a JuMP/NLPModels interface via Fortran; ExaCUTEst provides hand-written (or generated) ExaModels implementations that can run on GPU backends.

## How to add a new problem

1. Create `src/<SIFNAME>.jl` with a function `<lowercase>_model(; T = Float64, backend = nothing, kwargs...)`.
2. Include it in `src/ExaCUTEst.jl` and add to the `export` list.
3. Add a test in `test/runtests.jl` that compares `obj`, `grad`, and `cons` against `CUTEstModel("<SIFNAME>")`.

See `src/ROSENBR.jl` as the simplest example, `src/TAME.jl` for a problem with equality constraints, and `src/DEGDIAG.jl` for a large scalable problem.

## SIF format — key semantics

Understanding the SIF format is essential for converting problems correctly.

### Group structure

Every SIF group defines either an objective term (N type) or a constraint (E/L/G type). The group value is:

```
v_g = linear_part(x) + element_contributions(x) - constant_rhs
```

If a `'SCALE'` σ is given, the **group function** receives `v_g / σ` and the result is multiplied by `σ`:

```
contribution = σ * φ(v_g / σ)
```

For GROUP TYPE `L2` (φ(u) = u²): `contribution = σ * (v/σ)² = v²/σ`.

Examples:
- ROSENBR G1: σ=0.01, v = x2 − x1²  →  contribution = (x2−x1²)²/0.01 = **100(x2−x1²)²** ✓
- HS3 G2: σ=100000, v = x2−x1  →  contribution = (x2−x1)²/100000 = **10⁻⁵(x2−x1)²** ✓

### Variable bounds default

When no BOUNDS section is present for a variable, SIF defaults to **x ≥ 0** (not free). A `FR` bound makes a variable free. Always set `lvar` accordingly in ExaModels.

### Constraint convention

CUTEst.jl returns constraint values as `g(x) − b` (residual form), i.e., `cons(model, x) = 0` at feasibility. ExaModels uses `lcon ≤ expr ≤ ucon`, so write:
```julia
ExaModels.@add_con(c, linear_part(x) + nonlinear_elements(x) - rhs)
# lcon = 0.0, ucon = 0.0  (default for equality)
```
For inequality groups (L type: ≤ 0, G type: ≥ 0), set `lcon`/`ucon` accordingly.

### GROUP TYPE application

| SIF type | φ(v) |
|----------|------|
| L2 / SQUARE | v² |
| SINE | sin(v) |
| COS | cos(v) |
| EXP / EXPN | exp(v) |

Without a GROUP TYPE, the group function is the identity (φ(v) = v), i.e., a pure linear/element sum.

### ELEMENT TYPE

Element types define nonlinear functions of a subset of variables. The element value is added (with a coefficient) to the group's nonlinear part. Common elements:
- `SQ(V1)` = V1²  (coefficient in GROUP USES is the multiplier)

### QUADRATIC section

Entries `X X(I) X(J) coeff` add **0.5 * coeff * x_i * x_j** to the objective (Hessian convention). A diagonal entry `X(I) X(I) 1.0` → **0.5 * x_i²**, not **x_i²**.

### DO/OD loops and parameter arithmetic

Scalable SIF problems use DO/OD loops with integer parameters (IE) and real arithmetic (RE, RI, RA, R*). These must be replicated in Julia — see `src/DEGDIAG.jl` for an example with `lvar = T[i/(N+1) for i in 0:N]`.

## Testing

```julia
julia --project=. test/runtests.jl
```

Each test loads both the ExaCUTEst model and the reference `CUTEstModel` and compares `nvar`, `ncon`, `x0`, `lvar`, `uvar`, `obj`, `grad`, and `cons` at the starting point.

## SIF features and conversion difficulty

| Feature | Difficulty | Examples |
|---------|-----------|---------|
| Fixed variables + simple group types | Easy | ROSENBR, HS5, TAME, ALSOTAME |
| Scalable DO/OD loops | Medium | DEGDIAG, GOFFIN |
| Element types (multi-variable) | Medium | HS1, ROSENBR |
| Parameter arithmetic (RI/RA/R/) | Medium | DEGDIAG bounds |
| TEMPORARIES in elements/groups | Hard | ALSOTAME-style |
| QUADRATIC section | Medium | DEGDIAG |
| RANGES for inequality constraints | Easy | most HS problems |
