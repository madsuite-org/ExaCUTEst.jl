# HS88, HS89, HS90, HS91, HS92: a generic Julia port of their external Fortran
# (`EVAL88` ... `EVAL92`).
#
# The five SIF files are identical apart from the number of variables
# N = 2, 3, 4, 5, 6 (and the commented-out reference solutions) -- one routine
# shape, one port, one builder.
#
# WHAT THE FORTRAN COMPUTES
# -------------------------
# A time-optimal heat-conduction problem (Hock & Schittkowski 88-92). Each SIF
# declares
#
#     OBJ :  sum_{i=1}^{N} x_i^2                       (N SQR elements)
#     CON :  G-type row, constant -EPS^2 = -1e-4,
#            holding the single external element H with weight -1.0
#
# so the only thing the Fortran provides is the element value H(x) = EVAL(x);
# the objective is ordinary decoded SIF.
#
# EVAL(x) is a quadratic form in 30 "rho" values:
#
#     p_i      = sum_{k=i}^{N} x_k^2                        (i = 1..N)
#     u_j      = exp(-m_j p_1) + sum_{i=2}^{N} c_i exp(-m_j p_i) + (-1)^N
#     rho_j    = -u_j / m_j,          m_j = mu_j^2,  c_i = 2 (-1)^{i-1}
#     EVAL(x)  = 2/15 + sum_j s_j rho_j + sum_{i,j} R_ij rho_i rho_j
#
# The alternating coefficients come straight from the Fortran's `ALPHA`, which
# starts at -2.0 and is negated once per loop pass; the trailing `(-1)^N` is
# its `U = U + 0.5D0 * ALPHA` after the loop, with ALPHA then equal to
# -2(-1)^{N-1}.
#
# mu_1..mu_30 are a DATA table in the file (the roots of mu tan(mu) = 1, to
# double precision), and s_j and R_ij are "integration constants" the Fortran
# recomputes on every call from that table alone -- they do not depend on x:
#
#     a_i  = 2 sin(mu_i) / (mu_i + sin(mu_i) cos(mu_i))
#     s_i  = 2 a_i (cos(mu_i) - sin(mu_i)/mu_i)
#     R_ij = 0.5 (sin(mu_i+mu_j)/(mu_i+mu_j) + sin(mu_i-mu_j)/(mu_i-mu_j))
#              * a_i mu_i^2 * a_j mu_j^2                       (i != j)
#     R_ii = 0.5 (1 + 0.5 sin(2 mu_i)/mu_i) (a_i mu_i^2)^2
#
# They are hoisted into `hs88_constants(T)` and built once per element type.
#
# THE SHAPE, AND WHY
# ------------------
# EVAL depends on the VARIABLES, so it cannot be constant-folded the way
# VANDANIUM's B-spline is. It is therefore a plain generic Julia function
# (`hs88_eval`) interfaced to ExaModels as a `VectorNonlinearOracle`, with the
# gradient and Hessian taken by ForwardDiff over that same function.
#
# It is NOT an ExaModels symbolic expression, and that is not a close call:
# one rho_j alone is ~(N+1) exponentials over a reverse cumulative sum of
# squares, and the quadratic form needs two of them per term, so the chain is
# 50-100 operations deep for N = 6. ExaModels nests a type parameter per
# operation; a ~30-operation chain already failed to finish compiling in an
# hour elsewhere in this package. Written as ordinary Julia it compiles
# instantly, evaluates in Float64/Float32/BigFloat with no second code path,
# and is allocation- and branch-free on values, so it can be called from a GPU
# kernel later.
#
# The Fortran's hand-coded G(N) and H(N,N) arrays are deliberately NOT ported.
# Only the VALUE is translated; every derivative comes from AD over it. That is
# the error class that made the legacy HS67 port wrong in the fourth digit.
#
# The objective side stays decoded SIF: the N SQR elements are read out of the
# decode and emitted as one SIMD ExaModels generator, so only the constraint
# row goes through the oracle.
#
# VERIFIED 2026-10-16 against a real CUTEstModel of each problem, at x0 and at
# a perturbed point, permuted by SIF variable/constraint name: obj, grad, cons,
# jac, hprod (three random directions per point) and the FULL Lagrangian
# Hessian, plus lcon/ucon. Worst relative error over all five problems and all
# callbacks: 6.0e-16, i.e. rounding. N <= 6 here, so the full Hessian is
# materializable and no wrong row can hide orthogonal to the random hvp probes
# -- a gap that necessarily remains for BA-L*.
# Script: ~/git/scratch/hs88_92_verify.jl, results hs88_92_verify.tsv.
#
# NOT MATCHED, and upstream: the model cannot be built with T != Float64.
# ExaModels' _build_with_oracle allocates `work_ncon`/`work_nvar` as `zeros(n)`
# (Float64) while `ExaModelWithOracle` declares them `::VT`, so NO
# oracle-backed ExaModel builds in Float32 -- BA-L* included. The port itself
# is fine in Float32 and BigFloat (checked directly: Float64 agrees with
# BigFloat to 3e-16, Float32 to ~4e-07), so this is a one-line fix in
# ExaModels, not a limitation here.

"""
    HS88_MU :: NTuple{30,Float64}

The `DATA MU` table shared by HS88-HS92: the first thirty positive roots of
`mu tan(mu) = 1`, exactly as the Fortran spells them. Used as data, never
refined -- a BigFloat evaluation of the port is still an evaluation of *this*
problem, which is defined by these digits.
"""
const HS88_MU = (
    8.6033358901938017e-01, 3.4256184594817283e+00, 6.4372981791719468e+00,
    9.5293344053619631e+00, 1.2645287223856643e+01, 1.5771284874815882e+01,
    1.8902409956860023e+01, 2.2036496727938566e+01, 2.5172446326646664e+01,
    2.8309642854452012e+01, 3.1447714637546234e+01, 3.4586424215288922e+01,
    3.7725612827776501e+01, 4.0865170330488070e+01, 4.4005017920830845e+01,
    4.7145097736761031e+01, 5.0285366337773652e+01, 5.3425790477394663e+01,
    5.6566344279821521e+01, 5.9707007305335459e+01, 6.2847763194454451e+01,
    6.5988598698490392e+01, 6.9129502973895256e+01, 7.2270467060308960e+01,
    7.5411483488848148e+01, 7.8552545984242926e+01, 8.1693649235601683e+01,
    8.4834788718042290e+01, 8.7975960552493220e+01, 9.1117161394464745e+01,
)

"Length of the `MU` table; a compile-time constant so `ntuple` stays typed."
const HS88_NMU = length(HS88_MU)

"""
    HS88Constants{T}

The x-independent part of `EVAL88`..`EVAL92`: `t0 = 2/15`, `mu2[j] = mu_j^2`,
the quadrature weights `s[j]`, and the coupling matrix `r[i,j]`. The Fortran
rebuilds these on every call; here they are built once by
[`hs88_constants`](@ref).
"""
struct HS88Constants{T}
    t0::T
    mu2::Vector{T}
    s::Vector{T}
    r::Matrix{T}
end

"""
    hs88_constants(T) -> HS88Constants{T}

Build the integration constants in element type `T`, reproducing the Fortran's
loops 10/20 term for term.

The arithmetic is done in `promote_type(T, Float64)` and then rounded to `T`,
because the Fortran is double precision throughout: for `T = Float64` that is
bit-for-bit the Fortran's own computation, for `T = Float32` it avoids
evaluating `sin(mu_i - mu_j)/(mu_i - mu_j)` at `mu ~ 91` in single precision
(where cancellation in the difference would dominate the answer), and for
`T = BigFloat` it widens properly rather than freezing the constants at
double.
"""
function hs88_constants(::Type{T}) where {T}
    W = promote_type(T, Float64)
    nmu = length(HS88_MU)
    mu = W[W(m) for m in HS88_MU]
    a = Vector{W}(undef, nmu)
    s = Vector{W}(undef, nmu)
    r = Matrix{W}(undef, nmu, nmu)
    for i in 1:nmu
        mui = mu[i]
        smui, cmui = sin(mui), cos(mui)
        ai = 2 * smui / (mui + smui * cmui)
        a[i] = ai
        s[i] = 2 * ai * (cmui - smui / mui)
        aimui2 = ai * mui^2
        for j in 1:i
            if i != j
                muj = mu[j]
                rij = (one(W) / 2) * (sin(mui + muj) / (mui + muj) +
                                      sin(mui - muj) / (mui - muj)) *
                      aimui2 * a[j] * muj^2
                r[i, j] = rij
                r[j, i] = rij
            else
                r[i, i] = (one(W) / 2) *
                          (one(W) + (one(W) / 2) * sin(mui + mui) / mui) * aimui2^2
            end
        end
    end
    return HS88Constants{T}(T(W(2) / W(15)), T[T(m^2) for m in mu],
                            T[T(v) for v in s], T[T(v) for v in r])
end

const _HS88_CONSTANTS_F64 = hs88_constants(Float64)

"""
    hs88_cached_constants(T) -> HS88Constants

The constants for element type `T`, reusing the module-level `Float64` table
when that is what is asked for.

`T` is reduced through `ForwardDiff.Dual` first: the constants are data, so a
`Dual` needs the constants of its value type and no partials. Without that
reduction the table would be rebuilt in `Dual` arithmetic on every AD call --
and, worse, a `Float32` model would silently promote the whole evaluation to
`Float64`, because the constants would carry the wider type.
"""
hs88_cached_constants(::Type{T}) where {T} = _hs88_constants_for(_hs88_base(T))

_hs88_base(::Type{T}) where {T} = T
_hs88_base(::Type{D}) where {D<:ForwardDiff.Dual} = _hs88_base(ForwardDiff.valtype(D))

_hs88_constants_for(::Type{Float64}) = _HS88_CONSTANTS_F64
_hs88_constants_for(::Type{T}) where {T} = hs88_constants(T)

"""
    hs88_cumsq(x::NTuple{N}) -> NTuple{N}

`p_i = sum_{k=i}^{N} x_k^2`, the Fortran's loop 100 (`P(K) = P(K+1) + X(K)**2`
run downwards). Written as a tuple recursion so it unrolls and allocates
nothing.
"""
@inline hs88_cumsq(x::NTuple{1,V}) where {V} = (x[1] * x[1],)
@inline function hs88_cumsq(x::NTuple{N,V}) where {N,V}
    t = hs88_cumsq(Base.tail(x))
    return (t[1] + x[1] * x[1], t...)
end

"""
    hs88_rho(p::NTuple{N}, mu2) -> value

One `rho_j`, from the reverse cumulative sums `p` and `mu2 = mu_j^2`:

    u = exp(-mu2 p_1) + sum_{i=2}^{N} alpha_i exp(-mu2 p_i) + alpha_end/2
    rho = -u / mu2

with `alpha` starting at `-2` and negated each pass, exactly as the Fortran's
loop 180 does. No branch on a value, no allocation; generic in the element
type, so it takes `Dual`s for AD and `BigFloat` for a precision check.
"""
@inline function hs88_rho(p::NTuple{N,V}, mu2::T) where {N,V,T}
    R = promote_type(V, T)
    u = exp(-mu2 * p[1])
    alpha = -R(2)
    for i in 2:N
        u += alpha * exp(-mu2 * p[i])
        alpha = -alpha
    end
    u += alpha / 2
    return -u / mu2
end

"""
    hs88_eval(x::NTuple{N}, c::HS88Constants = hs88_cached_constants(...)) -> value

The value of the HS88-HS92 external element: `EVAL(x)` as the Fortran's
`EVAL88`..`EVAL92` compute it, for any `N`.

    EVAL = t0 + sum_j s_j rho_j + sum_{i,j} R_ij rho_i rho_j

This is the whole port. Generic in the element type, allocation-free and free
of branches on values; the only derivative route is AD over it.
"""
function hs88_eval(x::NTuple{N,V}, c::HS88Constants{T}) where {N,V,T}
    R = promote_type(V, T)
    p = hs88_cumsq(x)
    rho = ntuple(j -> R(hs88_rho(p, c.mu2[j])), Val(HS88_NMU))
    acc = R(c.t0)
    for i in 1:HS88_NMU
        t = R(c.s[i])
        for j in 1:HS88_NMU
            t += R(c.r[i, j]) * rho[j]
        end
        acc += rho[i] * t
    end
    return acc
end

hs88_eval(x::NTuple{N,V}) where {N,V} = hs88_eval(x, hs88_cached_constants(V))

"""
    hs88_eval(x::AbstractVector) -> value

Convenience wrapper for testing the port directly on a vector of values. Not
type-stable in `N`; the tuple form is the one the model uses.
"""
hs88_eval(x::AbstractVector{V}) where {V} =
    hs88_eval(NTuple{length(x),V}(x), hs88_cached_constants(V))

# ---------------------------------------------------------------------------
# Wiring: the generic function as an ExaModels external oracle.
#
# Everything below is assembly. The row is
#     w/s * EVAL(x_sub) - constant/s
# with w = -1, s = 1 and constant = -EPS^2 read out of the decode rather than
# hardcoded, and bounds 0 <= row < Inf from the G-group rule in
# ExaModelsSIF's builder. The element touches every variable, so the Jacobian
# row is dense and the Hessian block is the full lower triangle.

"""
    hs88_row_value(x, idx::NTuple{N,Int}, w, cst, c) -> value

The CON row's value at the full variable vector `x`: the port, scaled by the
group's element weight, with the group constant folded in.
"""
@inline function hs88_row_value(x, idx::NTuple{N,Int}, w, cst,
                                c::HS88Constants) where {N}
    v = ntuple(k -> x[idx[k]], N)
    return w * hs88_eval(v, c) + cst
end

"Gradient of the CON row w.r.t. its N variables, by AD over `hs88_eval`."
function hs88_row_grad(x, idx::NTuple{N,Int}, w, c::HS88Constants) where {N}
    v = [x[idx[k]] for k in 1:N]
    g = ForwardDiff.gradient(z -> hs88_eval(NTuple{N,eltype(z)}(z), c), v)
    return w .* g
end

"Hessian (N x N) of the CON row w.r.t. its N variables, by AD over `hs88_eval`."
function hs88_row_hess(x, idx::NTuple{N,Int}, w, c::HS88Constants) where {N}
    v = [x[idx[k]] for k in 1:N]
    H = ForwardDiff.hessian(z -> hs88_eval(NTuple{N,eltype(z)}(z), c), v)
    return w .* H
end

# Built behind a Val barrier so the closures below see a concrete N.
function _hs88_constraint!(core, ::Val{N}, idx::NTuple{N,Int}, nvar::Int,
                           w::T, cst::T, lcon::T, ucon::T) where {N,T}
    c = hs88_cached_constants(T)
    jr = ones(Int, N)
    jc = Int[idx[k] for k in 1:N]
    nh = (N * (N + 1)) ÷ 2
    hr = Vector{Int}(undef, nh)
    hc = Vector{Int}(undef, nh)
    let q = 0
        for a in 1:N, b in 1:a
            q += 1
            i, j = idx[a], idx[b]
            hr[q], hc[q] = max(i, j), min(i, j)
        end
    end
    vor = ExaModels.VectorNonlinearOracle(;
        nvar = nvar, ncon = 1,
        jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
        lcon = T[lcon], ucon = T[ucon],
        f! = (cv, xv) -> begin
            cv[1] = hs88_row_value(xv, idx, w, cst, c)
            cv
        end,
        jac! = (vals, xv) -> begin
            g = hs88_row_grad(xv, idx, w, c)
            for a in 1:N
                vals[a] = g[a]
            end
            vals
        end,
        hess! = (vals, xv, y) -> begin
            H = hs88_row_hess(xv, idx, w, c)
            y1 = y[1]
            q = 0
            for a in 1:N, b in 1:a
                vals[q += 1] = y1 * H[a, b]
            end
            vals
        end,
        adapt = Val(true))
    return ExaModels.constraint(core, vor)
end

"""
    hs88_92_model(name; T = Float64, backend = nothing) -> ExaModel

Build one of HS88, HS89, HS90, HS91, HS92 with the external Fortran replaced
by [`hs88_eval`](@ref).

The objective (`sum x_i^2`) is read out of `sif_decode(...; external =
:defer)` and emitted as a SIMD ExaModels generator; the single `G` row, whose
only content is the external element, becomes a `VectorNonlinearOracle` over
the port. The structure is checked against the decode rather than assumed, so
a changed SIF errors instead of quietly building the wrong model.
"""
function hs88_92_model(name::AbstractString; T = Float64, backend = nothing,
                      kwargs...)
    # The oracle callbacks are CPU Julia. hs88_eval itself is allocation- and
    # branch-free and therefore GPU-safe, but calling it from a device kernel
    # is not wired yet, and accepting `backend` here would misreport the model
    # as living on the device.
    backend === nothing || error(
        "hs88_92_model does not take a backend yet: the port's oracle " *
        "callbacks run on the CPU. hs88_eval is GPU-safe, so this is a " *
        "wiring step, not a limitation of the port.")

    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    length(P.external_types) == 1 && haskey(P.external_types, "H") ||
        error("$name: expected exactly one deferred element type H, found " *
              "$(collect(keys(P.external_types)))")
    nvar = length(P.varnames)

    objnames = [g for g in P.grouporder if P.groups[g].kind == 'N']
    connames = [g for g in P.grouporder if P.groups[g].kind != 'N']
    length(objnames) == 1 && length(connames) == 1 ||
        error("$name: expected 1 objective group and 1 constraint group, got " *
              "$(length(objnames)) and $(length(connames))")

    core = ExaModels.ExaCore(T)
    core, x = ExaModels.add_var(core, nvar; start = P.x0,
                                lvar = P.lvar, uvar = P.uvar)

    # --- objective: the decoded SQR elements, as data for one SIMD kernel
    og = P.groups[objnames[1]]
    isempty(og.lin) && og.constant == 0.0 && og.range_ === nothing ||
        error("$name: objective group $(objnames[1]) is not a plain sum of elements")
    odata = NamedTuple{(:j, :a),Tuple{Int,T}}[]
    for (en, w) in og.elems
        e = P.elements[en]
        e.etype == "SQR" ||
            error("$name: objective holds a $(e.etype) element; expected SQR")
        push!(odata, (j = e.varmap[:X], a = T(w / og.scale)))
    end
    isempty(odata) && error("$name: objective group has no elements")
    core, _ = ExaModels.add_obj(core,
                                Base.Generator(d -> d.a * x[d.j] * x[d.j], odata))

    # --- constraint: the external element, as an oracle over the port
    g = P.groups[connames[1]]
    g.kind == 'G' || error("$name: expected a G-type constraint, got $(g.kind)")
    g.range_ === nothing ||
        error("$name: constraint $(connames[1]) carries a range; unexpected")
    isempty(g.lin) ||
        error("$name: constraint $(connames[1]) has a linear part; unexpected")
    length(g.elems) == 1 ||
        error("$name: constraint $(connames[1]) holds $(length(g.elems)) " *
              "elements; expected exactly 1")
    en, w = g.elems[1]
    e = P.elements[en]
    e.etype == "H" || error("$name: constraint element is a $(e.etype)")
    evars = P.eltypes["H"].evars
    length(evars) == nvar ||
        error("$name: element H has $(length(evars)) elemental variables but " *
              "the problem has $nvar")
    idx = Tuple(e.varmap[v] for v in evars)

    s = g.scale
    # Same rule as ExaModelsSIF's builder: the constant is folded into the row
    # value and a G row is bounded below by zero.
    core = _hs88_constraint!(core, Val(nvar), idx, nvar,
                             T(w / s), T(-g.constant / s), zero(T), T(Inf))
    return ExaModels.ExaModel(core; kwargs...)
end

"""
    hs88_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS88 (2 variables) through [`hs88_92_model`](@ref).
"""
hs88_model(; kwargs...) = hs88_92_model("HS88"; kwargs...)

"""
    hs89_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS89 (3 variables) through [`hs88_92_model`](@ref).
"""
hs89_model(; kwargs...) = hs88_92_model("HS89"; kwargs...)

"""
    hs90_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS90 (4 variables) through [`hs88_92_model`](@ref).
"""
hs90_model(; kwargs...) = hs88_92_model("HS90"; kwargs...)

"""
    hs91_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS91 (5 variables) through [`hs88_92_model`](@ref).
"""
hs91_model(; kwargs...) = hs88_92_model("HS91"; kwargs...)

"""
    hs92_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS92 (6 variables) through [`hs88_92_model`](@ref).
"""
hs92_model(; kwargs...) = hs88_92_model("HS92"; kwargs...)
