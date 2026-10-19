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
# and is allocation- and branch-free on values, which is what lets a GPU kernel
# call it.
#
# TWO BUILDERS, ONE VALUE FUNCTION. `hs88_92_model` interfaces `hs88_eval`
# through host loops; `hs88_92_kernel_model` (end of this file) interfaces the
# same function through KernelAbstractions kernels and so accepts a `backend`.
# The host loops are what was verified against the Fortran, and the kernels are
# verified against them.
#
# The Fortran's hand-coded G(N) and H(N,N) arrays are deliberately NOT ported.
# Only the VALUE is translated; every derivative comes from AD over it. That is
# the error class that made the legacy HS67 port wrong in the fourth digit.
#
# The objective side stays decoded SIF: the N SQR elements are read out of the
# decode and emitted as one SIMD ExaModels generator, so only the constraint
# row goes through the oracle.
#
# Verified against a real CUTEstModel of each problem, at x0 and at
# a perturbed point, permuted by SIF variable/constraint name: obj, grad, cons,
# jac, hprod (three random directions per point) and the FULL Lagrangian
# Hessian, plus lcon/ucon. Worst relative error over all five problems and all
# callbacks: 6.0e-16, i.e. rounding. N <= 6 here, so the full Hessian is
# materializable and no wrong row can hide orthogonal to the random hvp probes
# -- a gap that necessarily remains for BA-L*.
# `test/native_verify.jl` re-runs the comparison. `hs88_92_kernel_model` is
# compared against THIS builder rather than against the Fortran a second time,
# which is what puts it on the same chain (`test/hs_gpu_verify.jl`).
#
# PRECISION. These models build and evaluate at any `T`: Float32 and BigFloat
# both return their own element type from every callback, and the port's
# arithmetic is generic (Float64 agrees with BigFloat to 3e-16 and Float32 to
# ~4e-7 on the element function itself). The DERIVATIVES carry that precision
# too, and both builders do. The reading that shows it, at a point chosen so
# that no coordinate is exactly a Float64: a BigFloat model's `jac` equals a
# BigFloat ForwardDiff gradient of `hs88_eval` to the last bit at 256-bit
# precision, and none of its entries is representable as a Float64.
#
# That rests on the pinned ExaModels revision, which allocates an oracle
# cache's callback buffers in the model's element type. A revision that
# allocates them as Float64 hands the callback a Float64 output array, and a
# BigFloat model then arbitrates Float64 to about one ulp and no finer --
# silently, since every callback still returns BigFloat values.

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

The array fields are typed only as `AbstractVector{T}` / `AbstractMatrix{T}`,
so the same struct holds host arrays or a device backend's arrays; a kernel
reassembles it from its own arguments, which is why [`hs88_eval`](@ref) serves
the host and the device unchanged.
"""
struct HS88Constants{T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    t0::T
    mu2::V
    s::V
    r::M
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
    return HS88Constants(T(W(2) / W(15)), T[T(m^2) for m in mu],
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
    # This builder's oracle callbacks are host loops, which cannot index a
    # device array; accepting `backend` here would misreport the model as
    # living on the device. `hs88_92_kernel_model` is the one that takes a
    # backend.
    backend === nothing || error(
        "hs88_92_model does not take a backend: its oracle callbacks are host " *
        "loops. Use hs88_92_kernel_model(name; backend = ...), which " *
        "evaluates the same hs88_eval from KernelAbstractions kernels.")

    p = _hs88_92_parts(name, T)
    core = ExaModels.ExaCore(T)
    core, x = ExaModels.add_var(core, p.nvar; start = p.x0,
                                lvar = p.lvar, uvar = p.uvar)
    core, _ = ExaModels.add_obj(core,
                                Base.Generator(d -> d.a * x[d.j] * x[d.j], p.odata))
    core = _hs88_constraint!(core, Val(p.nvar), p.idx, p.nvar, p.w, p.cst,
                             zero(T), T(Inf))
    return ExaModels.ExaModel(core; kwargs...)
end

"""
    _hs88_92_parts(name, ::Type{T}) -> NamedTuple

Decode `name`, check that it has the HS88-HS92 shape, and return what both
builders need: `nvar`, the `x0`/`lvar`/`uvar` vectors, the objective's `SQR`
elements as `(j, a)` data for one SIMD generator, the external element's
variable indices `idx`, and the constraint row's weight `w` and folded
constant `cst`.

The structure is checked against the decode rather than assumed, so a changed
SIF errors instead of quietly building the wrong model.
"""
function _hs88_92_parts(name::AbstractString, ::Type{T}) where {T}
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

    # --- constraint: the external element, whose value the port supplies
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

    # Same rule as ExaModelsSIF's builder: the group constant is folded into
    # the row value, and a G row is bounded below by zero.
    s = g.scale
    return (nvar = nvar, x0 = P.x0, lvar = P.lvar, uvar = P.uvar, odata = odata,
            idx = idx, w = T(w / s), cst = T(-g.constant / s))
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

# ---------------------------------------------------------------------------
# The same port called from KernelAbstractions kernels, with the MU-derived
# constants resident on the model's backend.
#
# WHAT IS PARALLEL: nothing. These problems have ONE constraint row over
# N <= 6 variables, so every kernel here runs with `ndrange = 1` and a single
# thread does the whole evaluation. The reason to have a kernel at all is that
# the row must be evaluable with its data on a device and scalar indexing
# disabled (`CUDA.allowscalar(false)`), which a host loop over a device array
# cannot be. It is not a throughput claim.
#
# The objective, `sum x_i^2`, is N independent terms and stays an ordinary SIMD
# ExaModels generator, which already runs on any backend.
#
# WHY THE KERNEL MODEL IS BUILT WITH `prod = true`. Mixing a SIMD symbolic part
# with an oracle is what makes it necessary: ExaModels' KernelAbstractions
# override of `hprod!` for a model with oracles dispatches on the prodhelper
# that `prod = true` builds, and without it the call falls back to the generic
# method, which evaluates the SIMD objective's Hessian through host loops over
# the device arrays -- `Scalar indexing is disallowed`. HS67 and HS85 have no
# SIMD part (their objective is an oracle too) and so do not need it.
#
# ONE BUILDER FOR BOTH BACKENDS. `backend = nothing` runs the same kernels on
# KernelAbstractions' CPU backend, so a CPU-versus-GPU comparison tests the
# device rather than two unrelated implementations; `test/hs_gpu_verify.jl` is
# that comparison, and it also compares the host kernels against
# [`hs88_92_model`](@ref), which is what was verified against the Fortran.
#
# THE CONSTANTS REACH THE DEVICE AS ARRAYS. `mu2`, `s` and `r` are converted
# once at build time and passed as kernel arguments; the kernel rebuilds
# `HS88Constants` around them, so `hs88_eval` is the one called on both sides.
# Rebuilding the table inside the kernel would mean 30 sin/cos pairs and a
# 900-entry matrix per call, which is what the Fortran does and what
# [`hs88_constants`](@ref) exists to avoid.
#
# No `@inbounds`: the gathers go through `idx`, whose entries are checked once
# at build time (see `_hs88_check_indices`), and the oracle value buffers are
# written at the lengths ExaModels declared for them.
#
# ORACLE BUFFER INDEXING. ExaModels' oracle contract hands each callback a
# dense one-based buffer of the declared length (`cv[1:ncon]`,
# `vals[1:nnzj]`, `vals[1:nnzh]`, `Hv[1:nvar]`), so these kernels write it as
# such.

"""
    _hs88_check_indices(idx, nvar)

Check that the element's variable indices are a permutation of `1:nvar`.

Two kernel properties rest on this and on nothing else. The gathers
`x[idx[k]]` have no bounds check of their own, and on a device an out-of-range
gather has no useful diagnosis -- it reads whatever memory is there. And
`_hs88_hvp_kernel!` ASSIGNS rather than accumulates into `Hv[idx[a]]`, which
fills the whole buffer only because `idx` is onto `1:nvar`. The check runs once
per build.
"""
function _hs88_check_indices(idx::NTuple{N,Int}, nvar::Integer) where {N}
    N == nvar || error("element H has $N elemental variables but the problem " *
                       "has $nvar")
    seen = falses(nvar)
    for (k, j) in pairs(idx)
        1 <= j <= nvar ||
            error("elemental variable $k maps to index $j, outside 1:$nvar")
        seen[j] && error("elemental variable $k repeats variable index $j")
        seen[j] = true
    end
    return nothing
end

@inline _hs88_gather(x, idx::NTuple{N,<:Integer}) where {N} =
    ntuple(k -> x[idx[k]], Val(N))

# ForwardDiff seeding written out rather than going through
# ForwardDiff.gradient/hessian: those take an `AbstractVector`, allocate a
# result and consult a mutable tag counter, none of which is available in a
# kernel. Seeding by hand keeps everything in registers.
@inline _hs88_seed(::Type{S}, ::Val{N}, i::Int) where {S,N} =
    ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(S), zero(S)), Val(N)))

"""
    _hs88_valgrad(v, c) -> (value, gradient::NTuple{N})

`hs88_eval` and its gradient in the N elemental variables, by one level of
forward-mode AD over the same function the host port differentiates.
"""
@inline function _hs88_valgrad(v::NTuple{N,T}, c::HS88Constants) where {N,T}
    D = ForwardDiff.Dual{:HS88G,T,N}
    vd = ntuple(i -> D(v[i], _hs88_seed(T, Val(N), i)), Val(N))
    r = hs88_eval(vd, c)
    p = ForwardDiff.partials(r)
    return ForwardDiff.value(r), ntuple(a -> p[a], Val(N))
end

"""
    _hs88_hess(v, c) -> NTuple{N,Partials{N}}

The N x N Hessian of `hs88_eval`, by nesting a second level of forward-mode
AD. Row `a` comes back as a `Partials{N}`; the matrix is symmetric, so callers
read whichever half they need.
"""
@inline function _hs88_hess(v::NTuple{N,T}, c::HS88Constants) where {N,T}
    D1 = ForwardDiff.Dual{:HS88H1,T,N}
    D2 = ForwardDiff.Dual{:HS88H2,D1,N}
    vd = ntuple(i -> D2(D1(v[i], _hs88_seed(T, Val(N), i)),
                        _hs88_seed(D1, Val(N), i)), Val(N))
    p = ForwardDiff.partials(hs88_eval(vd, c))
    return ntuple(a -> ForwardDiff.partials(p[a]), Val(N))
end

# --- kernels. `idx` is passed by value as a tuple, not as an array: N <= 6, so
# the indices live in registers and `Val(N)` comes out of the tuple's type.

@kernel function _hs88_cons_kernel!(cv, @Const(x), idx, @Const(mu2), @Const(s),
                                    @Const(r), t0, w, cst)
    i = @index(Global)
    c = HS88Constants(t0, mu2, s, r)
    cv[i] = w * hs88_eval(_hs88_gather(x, idx), c) + cst
end

@kernel function _hs88_jac_kernel!(vals, @Const(x), idx, @Const(mu2), @Const(s),
                                   @Const(r), t0, w)
    @index(Global)
    c = HS88Constants(t0, mu2, s, r)
    _, g = _hs88_valgrad(_hs88_gather(x, idx), c)
    for a in eachindex(g)
        vals[a] = w * g[a]
    end
end

# The row's Hessian contribution, y[1] * w * nabla^2 EVAL, as the lower
# triangle in the slot order the builder declares.
@kernel function _hs88_hess_kernel!(vals, @Const(x), idx, @Const(mu2), @Const(s),
                                    @Const(r), t0, w, @Const(mult))
    @index(Global)
    c = HS88Constants(t0, mu2, s, r)
    H = _hs88_hess(_hs88_gather(x, idx), c)
    sc = w * mult[1]
    q = 0
    for a in eachindex(H), b in 1:a
        q += 1
        vals[q] = sc * H[a][b]
    end
end

# (y[1] w nabla^2 EVAL) * v. One thread owns every entry of `Hv` it writes,
# because `idx` is a permutation of 1:nvar, so this assigns and needs no
# atomic and no preceding fill.
@kernel function _hs88_hvp_kernel!(Hv, @Const(x), idx, @Const(mu2), @Const(s),
                                   @Const(r), t0, w, @Const(mult), @Const(vv))
    @index(Global)
    c = HS88Constants(t0, mu2, s, r)
    H = _hs88_hess(_hs88_gather(x, idx), c)
    vl = _hs88_gather(vv, idx)
    sc = w * mult[1]
    for a in eachindex(H)
        acc = zero(eltype(Hv))
        for b in eachindex(vl)
            acc += H[a][b] * vl[b]
        end
        Hv[idx[a]] = sc * acc
    end
end

"""
    hs88_92_kernel_model(name; T = Float64, backend = nothing) -> ExaModel

Build one of HS88, HS89, HS90, HS91, HS92 with the external Fortran replaced by
[`hs88_eval`](@ref), evaluated by KernelAbstractions kernels with the
`MU`-derived constants resident on `backend`.

`backend = nothing` builds a host model that runs the same kernels on
KernelAbstractions' CPU backend; `backend = CUDABackend()` builds a device
model that never indexes a device array from the host, so it is usable with
`CUDA.allowscalar(false)`.

`prod = true` is the default because `hprod` on a device needs it here; see
the file header. [`hs88_92_model`](@ref) builds the same problem through host
loops and is the one verified against the Fortran directly.
"""
function hs88_92_kernel_model(name::AbstractString; T = Float64,
                              backend = nothing, prod::Bool = true, kwargs...)
    p = _hs88_92_parts(name, T)
    _hs88_check_indices(p.idx, p.nvar)
    cs = hs88_cached_constants(T)
    ka = backend === nothing ? KernelAbstractions.CPU() : backend
    cv = v -> ExaModels.convert_array(v, backend)
    mu2, sw, rr, t0 = cv(cs.mu2), cv(cs.s), cv(cs.r), cs.t0

    core = ExaModels.ExaCore(T; backend = backend)
    core, x = ExaModels.add_var(core, p.nvar; start = p.x0,
                                lvar = p.lvar, uvar = p.uvar)
    core, _ = ExaModels.add_obj(core,
                                Base.Generator(d -> d.a * x[d.j] * x[d.j], p.odata))
    core = _hs88_kernel_constraint!(core, ka, p.idx, p.nvar, p.w, p.cst,
                                    mu2, sw, rr, t0, zero(T), T(Inf))
    return ExaModels.ExaModel(core; prod = prod, kwargs...)
end

# Built behind a function barrier so the closures below see a concrete tuple
# length for `idx`, exactly as `_hs88_constraint!` does for the host builder.
function _hs88_kernel_constraint!(core, ka, idx::NTuple{N,Int}, nvar::Int,
                                  w::T, cst::T, mu2, sw, rr, t0,
                                  lcon::T, ucon::T) where {N,T}
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
    sync = () -> KernelAbstractions.synchronize(ka)
    vor = ExaModels.VectorNonlinearOracle(;
        nvar = nvar, ncon = 1,
        jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
        lcon = T[lcon], ucon = T[ucon],
        f! = (cv, xv) -> begin
            _hs88_cons_kernel!(ka)(cv, xv, idx, mu2, sw, rr, t0, w, cst;
                                   ndrange = 1)
            sync()
            return cv
        end,
        jac! = (vals, xv) -> begin
            _hs88_jac_kernel!(ka)(vals, xv, idx, mu2, sw, rr, t0, w; ndrange = 1)
            sync()
            return vals
        end,
        hess! = (vals, xv, yv) -> begin
            _hs88_hess_kernel!(ka)(vals, xv, idx, mu2, sw, rr, t0, w, yv;
                                   ndrange = 1)
            sync()
            return vals
        end,
        hvp! = (Hv, xv, yv, vv) -> begin
            _hs88_hvp_kernel!(ka)(Hv, xv, idx, mu2, sw, rr, t0, w, yv, vv;
                                  ndrange = 1)
            sync()
            return Hv
        end,
        adapt = Val(false))
    return ExaModels.constraint(core, vor)
end

"""
    hs88_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS88 (2 variables) through [`hs88_92_kernel_model`](@ref).
"""
hs88_kernel_model(; kwargs...) = hs88_92_kernel_model("HS88"; kwargs...)

"""
    hs89_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS89 (3 variables) through [`hs88_92_kernel_model`](@ref).
"""
hs89_kernel_model(; kwargs...) = hs88_92_kernel_model("HS89"; kwargs...)

"""
    hs90_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS90 (4 variables) through [`hs88_92_kernel_model`](@ref).
"""
hs90_kernel_model(; kwargs...) = hs88_92_kernel_model("HS90"; kwargs...)

"""
    hs91_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS91 (5 variables) through [`hs88_92_kernel_model`](@ref).
"""
hs91_kernel_model(; kwargs...) = hs88_92_kernel_model("HS91"; kwargs...)

"""
    hs92_kernel_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

HS92 (6 variables) through [`hs88_92_kernel_model`](@ref).
"""
hs92_kernel_model(; kwargs...) = hs88_92_kernel_model("HS92"; kwargs...)
