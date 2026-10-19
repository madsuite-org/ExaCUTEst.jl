# CAR2 / ORBIT2: a generic Julia port of the `COLLOC` external Fortran.
#
# WHAT THE FORTRAN IS. Both SIF files append the SAME driver routine,
# `DOUBLE PRECISION FUNCTION COLLOC(...)`, differing only in how many states
# (NX), controls (NU) and extra parameters it carries. COLLOC is one row of a
# Hermite-type collocation discretization of
#
#     dX/dT = F(X, U, Q, T)
#
# over one time interval [T, T1], where X is the state at T, XP the state at
# T1, U the (interval-constant) control and Q a free scalar -- here the total
# manoeuvre time, which is why the whole right-hand side is multiplied by it.
# Writing TOR = T1 - T, C1 = TOR/8, C2 = 3/(2 TOR):
#
#     F  = FSYS(X,  U, T,  Q)          # slope at the left end
#     FP = FSYS(XP, U, T1, Q)          # slope at the right end
#     XC(j)     = (X(j) + XP(j))/2 + C1 (F(j) - FP(j))      # midpoint state
#     FCPRED(j) = C2 (XP(j) - X(j)) - (F(j) + FP(j))/4      # predicted midpoint slope
#     FC = FSYS(XC, U, (T + T1)/2, Q)  # actual slope at the midpoint
#     COLLOC = FCPRED(ID) - FC(ID)
#
# i.e. the defect of the ID-th equation at the interval midpoint; the SIF makes
# one element, and one equality row, per (interval, ID) pair.
#
# Only `FSYS` differs between the two problems, so it is the port's one
# parameter:
#
#   * CAR2 (NX=4, NU=2): a point mass on a plane with linear friction,
#     F = Q*(x3, x4, u1 - f x3, u2 - f x4). The friction coefficient `f`
#     arrives as the element parameter COEFRI, so it is read from the decode
#     rather than hardcoded and a `-param COEFRI=` variant still works.
#   * ORBIT2 (NX=6, NU=3): low-thrust orbit transfer. Position/velocity split,
#     inverse-square gravity through r = (x1²+x2²+x3²)^1.5, and a mass that
#     burns off linearly, M = 1 - HM T Q. Its four scaling constants are
#     computed INSIDE the Fortran, not taken from the SIF data, so they are
#     recomputed here in exactly the Fortran's association order (the SIF
#     happens to define HG/HV/HF/HM too, in a different order of operations --
#     the element function uses its own, so that is what is ported).
#
# THE SHAPE: a generic Julia function behind an ExaModels ORACLE, because the
# routine depends on the VARIABLES and is deep. One ORBIT2 row with ID in 4..6
# is three FSYS evaluations, two of them through a `^1.5`, plus the midpoint
# state: about 75 elementary operations. An ExaModels expression nests one type
# parameter per operation, and a chain this long does not finish compiling --
# `src/native/BA.jl` hits the same wall at ~30 operations. CAR2 is shallower
# and is still a plain function: the two problems share one Fortran routine, so
# they share one implementation.
#
# Derivatives come from ForwardDiff over `colloc`. The Fortran's hand-coded
# DX/DXP/DU/DQ chain-rule block and FSYS's DFDX/DFDU/DFDQ arrays are NOT
# ported: a transcribed derivative is a second thing to keep correct, and it
# goes wrong in digits a casual comparison does not reach.
#
# THREE THINGS ABOUT THE REFERENCE, all measured, all worth knowing before
# comparing against CUTEst.
#
# 1. CUTEST'S HESSIAN FOR THESE PROBLEMS IS ZERO ON THE COLLOC ELEMENTS. The
#    SIF's `Ktyp` INDIVIDUALS block gives `F`, an `A F` alternative and one
#    `G` line per elemental variable -- and no `H` lines at all. SIFdecode
#    therefore takes the element's second derivatives to be zero, which the
#    classification string says out loud: `LOR1-RN-V-V`, whose fourth
#    character `1` means only first derivatives are provided. Measured on
#    CAR2: CUTEst's Lagrangian Hessian has 1998 nonzeros, exactly the `2 w`
#    diagonal of the 999 control pairs in the SQR elements, and nothing else.
#    AD over `colloc` gives the TRUE Hessian instead (9995 nonzeros on CAR2),
#    and `element_hessian = true` is the DEFAULT, for consistency across the
#    family rather than by preference. Exactly five problems in the collection
#    carry a `1` in that position -- BLEACHNG, CAR2, ORBIT2, SARO, SAROMM, all
#    five external-Fortran (grep `CLASSF.DB` inside the collection artifact) --
#    and `src/native/SARO.jl` passes `hess!` to its oracle unconditionally,
#    with no knob, against an equally empty reference. A deliberately
#    incomplete Hessian here would make two problems in that set of five
#    disagree with two others.
#    `element_hessian = false` reproduces CUTEst's Hessian exactly. Its use is
#    diagnostic: it isolates the second derivatives, so that a comparison
#    against the reference distinguishes "the Hessian differs, as declared"
#    from "something else differs too".
#    The true Hessian is adjudicated against the REFERENCE rather than against
#    our own Jacobian, which would only be self-consistency: CUTEst's first
#    derivatives are sound and only its `H` is missing, so a central difference
#    of the reference's own Lagrangian gradient `J_ref(x)'y` is an independent
#    check. See note 3.
#
# 2. THE REFERENCE KEEPS 11 SIGNIFICANT DIGITS OF THE INTERVAL ENDPOINTS, AND
#    C2 AMPLIFIES THAT. The time nodes are `T(i) = (i-1)/(M-1)` and reach the
#    element function as the parameters TN/TN1. SIFdecode writes element
#    parameters out in a D-format, and the copies the compiled Fortran reads
#    carry 11 significant decimal digits. `TOR = TN1 - TN` is a cancelling
#    subtraction of two numbers up to 1000x larger, so that truncation becomes
#    a relative error of up to ~9e-9 in `C2 = 3/(2 TOR)` -- and C2 is the
#    largest Jacobian entry in either problem.
#    Measured, not inferred: for a CAR2 ID=1 row the Fortran's derivative
#    w.r.t. X(1) is exactly -C2, so TOR is recoverable from CUTEst's own
#    Jacobian. Over all 999 such rows, the recovered TOR equals
#    `round(TN1, sigdigits = 11) - round(TN, sigdigits = 11)` with relative
#    error 0.0, and differs from exact SIF arithmetic by up to 8.99e-9;
#    9, 10, 12 and 13 significant digits all fail. With TN/TN1 so rounded,
#    `colloc` reproduces the Fortran's value for every one of CAR2's 3996 and
#    ORBIT2's 1794 rows at relative error 0.0, at x0 and at a perturbed point.
#    So the ~1e-8 gap is the reference's stored precision and nothing else.
#    `ref_param_rounding = true` reproduces it. The default is `false`: the
#    full-precision values are what the SIF says, and rounding element
#    parameters to SIFdecode's output format is a decision for ExaModelsSIF to
#    take for the whole collection, not a fudge inside one port.
#    The same truncation shows up once more, outside this port's reach: with
#    `ref_param_rounding = true` CAR2 agrees with the Fortran on every row at
#    relative error 0.0, and ORBIT2 agrees on all 1794 COLLOC rows at 0.0 but
#    keeps 2.7e-11 on row L1, whose group CONSTANT is the SIF-computed
#    RFRF = (RF/R0)^2 -- the reference's copy of that constant is likewise
#    truncated (-0.030905307000000 against -0.030905307028187). That row has
#    no external Fortran in it at all; it is the decoder's, not the port's.
#
# 3. THE TRUE HESSIAN CHECKS OUT AGAINST THE REFERENCE'S FIRST DERIVATIVES.
#    Comparing `H v` with `(J_ref(x + h v)'y - J_ref(x - h v)'y) / 2h`, where
#    nothing of ours enters the right-hand side: the worst relative difference
#    over x0 and a perturbed point is 4.5e-8 for CAR2 and 3.5e-8 for ORBIT2 at
#    h = 1e-5. Shrinking h makes it WORSE -- ~3e-7 at 1e-6, ~4e-6 at 1e-7 --
#    which is the round-off branch of a central difference (cancellation grows
#    as eps/h while truncation falls as h^2), i.e. the difference quotient is
#    the inaccurate side of that comparison, not the analytic Hessian. A wrong
#    Hessian would instead sit at a fixed relative error independent of h.
#
# PRECISION, and one limit that is ExaModels' rather than the port's. `colloc`
# is generic: evaluated directly it gives the same number in Float64, Float32
# and BigFloat (Float64 agrees with BigFloat to ~1e-16 relative on every
# equation index of both problems). Through the oracle, though, ExaModels'
# `EvalIndexCache` allocates its callback buffers as plain `Float64`, so a
# `T = BigFloat` model still round-trips every oracle result through double
# precision. The port is not what caps that.
#
# ASSEMBLY, and why it is an `add_eval` and not a `VectorNonlinearOracle`.
# Neither problem is external all the way through: besides the K rows (one
# COLLOC element each) there is a linear objective `Q`, the control-magnitude
# rows `|U|² <= 1`, and in ORBIT2 three terminal rows built from SQR/PROD
# elements. Those parts are ordinary SIF and stay ordinary ExaModels SIMD
# expressions. The constraint rows must stay in SIF declaration order, because
# `sif_constraint_names` reports that order from an independent decode -- and
# `ExaModels.constraint(core, ::VectorNonlinearOracle)` APPENDS its rows after
# all SIMD rows, which would silently permute them. `ExaModels.add_eval` fills
# rows that already exist, so the K rows keep their declared positions and the
# port contributes only the COLLOC term.

# --------------------------------------------------------------------- FSYS

"""
    CarODE(fric)

CAR2's `FSYS`: a point mass on a plane under a driving force `u` and linear
friction, in rescaled time, so the whole field is multiplied by the free time
`q`. State `(x1, x2)` is position, `(x3, x4)` velocity. No explicit time
dependence. Callable as `sys(x, u, t, q)`, returning the four slopes.
"""
struct CarODE{R<:Real}
    fric::R
end

@inline function (s::CarODE)(x::NTuple{4,Any}, u::NTuple{2,Any}, t, q)
    return (q * x[3],
            q * x[4],
            q * (u[1] - s.fric * x[3]),
            q * (u[2] - s.fric * x[4]))
end

"""
    OrbitODE(hm, hv, hg, hf, p32)

ORBIT2's `FSYS`: a spacecraft under inverse-square gravity and a thrust `u`,
in rescaled time, with the propellant mass burning off linearly. State
`(x1, x2, x3)` is position, `(x4, x5, x6)` velocity;

    M = 1 - hm t q,   S = x1² + x2² + x3²,   R = S^p32

    F = q * (hv x4, hv x5, hv x6,
             -hg x1 / R - hf u1 / M,
             -hg x2 / R - hf u2 / M,
             -hg x3 / R - hf u3 / M)

`p32` is 1.5 held as a plain `Real` rather than a literal, so `S^p32` keeps
the working precision under `Float32`/`BigFloat` and still takes ForwardDiff's
power rule (an exponent promoted to a `Dual` would route through `exp∘log`).
"""
struct OrbitODE{R<:Real}
    hm::R
    hv::R
    hg::R
    hf::R
    p32::R
end

"""
    orbit2_ode(T = Float64)

The ORBIT2 `FSYS` with the scaling constants the Fortran computes for itself:
earth radius 6371 km, `MU = 9.81e-3 RT²`, initial orbit `R0 = RT + 150`,
`VORB = sqrt(MU/R0)`, spacecraft mass 3000 kg, `QT = 3.333`, `VG = 3.0`,
time scale 1. Written in the Fortran's association order.
"""
function orbit2_ode(::Type{T} = Float64) where {T}
    rt = T(6371)
    mu = T(981) / T(100000) * rt * rt     # 9.81D-3 * RT * RT
    r0 = rt + T(150)
    vorb = sqrt(mu / r0)
    m0 = T(3000)
    qt = T(3333) / T(1000)
    vg = T(3)
    ts = one(T)
    hm = ts * qt / m0
    hv = ts * vorb / r0
    hg = ts * mu / (vorb * r0 * r0)       # R0 ** 2 is an integer power
    hf = ts * qt * vg / (vorb * m0)
    return OrbitODE(hm, hv, hg, hf, T(3) / T(2))
end

@inline function (s::OrbitODE)(x::NTuple{6,Any}, u::NTuple{3,Any}, t, q)
    mt = s.hm * t * q
    m = one(mt) - mt
    ss = x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
    r = ss^s.p32
    return (s.hv * q * x[4],
            s.hv * q * x[5],
            s.hv * q * x[6],
            q * (-s.hg * x[1] / r - s.hf * u[1] / m),
            q * (-s.hg * x[2] / r - s.hf * u[2] / m),
            q * (-s.hg * x[3] / r - s.hf * u[3] / m))
end

# ------------------------------------------------------------------- COLLOC

"""
    colloc(sys, x, xp, u, q, t, t1, id) -> value

The value of the external `COLLOC`: the defect of equation `id` of
`dX/dT = sys(X, U, T, Q)` at the midpoint of `[t, t1]`, given the state `x` at
`t` and `xp` at `t1`.

`x`, `xp` and `u` are tuples; `id` is a build-time index into them, so there is
no branch on a traced value. Generic in the element type -- it evaluates on
`Float64`, on `ForwardDiff.Dual` for derivatives, and on `BigFloat` -- and
allocation-free.

Returns the VALUE only. The Fortran's `DERIV = .TRUE.` branch, which fills
`GX*/GXP*/GU*/GQ*` from `FSYS`'s hand-coded Jacobians, is deliberately not
ported; AD over this function supplies those.
"""
@inline function colloc(sys, x::NTuple{NX,Any}, xp::NTuple{NX,Any},
                        u::NTuple{NU,Any}, q, t, t1, id::Int) where {NX,NU}
    tor = t1 - t
    c1 = tor / 8                         # EIGHTH * TOR
    c2 = oftype(tor, 3) / (tor + tor)    # ONEPT5 / TOR
    tc = (t1 + t) / 2                    # HALF * ( T1 + T )
    f = sys(x, u, t, q)
    fp = sys(xp, u, t1, q)
    xc = ntuple(j -> (x[j] + xp[j]) / 2 + c1 * (f[j] - fp[j]), Val(NX))
    fc = sys(xc, u, tc, q)
    fcpred = c2 * (xp[id] - x[id]) - (f[id] + fp[id]) / 4
    return fcpred - fc[id]
end

# ------------------------------------------------- one SIF element instance

"""
    CollocElem{NX,NU,R}

One `Ktyp` element: where its `2*NX + NU + 1` elemental variables live in the
problem's variable vector, the interval endpoints `t`/`t1`, the equation index
`id`, the constraint row it lands in and the weight (element weight divided by
the group's scale) with which it enters that row.
"""
struct CollocElem{NX,NU,R<:Real}
    xi::NTuple{NX,Int}
    xf::NTuple{NX,Int}
    uv::NTuple{NU,Int}
    qv::Int
    t::R
    t1::R
    id::Int
    row::Int
    w::R
end

"Number of variables one element touches."
@inline collocnv(::CollocElem{NX,NU}) where {NX,NU} = 2 * NX + NU + 1

"The global index of this element's `k`-th local variable slot."
@inline function collocvar(e::CollocElem{NX,NU}, k::Int) where {NX,NU}
    k <= NX && return e.xi[k]
    k <= 2NX && return e.xf[k-NX]
    k <= 2NX + NU && return e.uv[k-2NX]
    return e.qv
end

"""
    CollocRowFun(sys, e)

`z -> colloc(...)` on the element's own `2*NX + NU + 1` variables, as a
callable struct rather than a closure: every element of a problem then shares
one concrete type, so a single `ForwardDiff` config (and hence one AD tag)
serves them all.
"""
struct CollocRowFun{S,E<:CollocElem}
    sys::S
    e::E
end

@inline function (f::CollocRowFun{S,CollocElem{NX,NU,R}})(z) where {S,NX,NU,R}
    e = f.e
    x = ntuple(k -> z[k], Val(NX))
    xp = ntuple(k -> z[NX+k], Val(NX))
    u = ntuple(k -> z[2NX+k], Val(NU))
    return colloc(f.sys, x, xp, u, z[2NX+NU+1], e.t, e.t1, e.id)
end

"""
    collocgather!(z, x, e)

Gather element `e`'s variables out of the full variable vector into `z`.

`k` runs over the element's local variable SLOTS, not over an axis of either
array: slot `k` reads `x[collocvar(e, k)]`, and those indices come from the
SIF decode as absolute 1-based positions. `require_one_based_indexing` makes
that assumption fail loudly on an offset `x` instead of silently reading the
wrong variables.
"""
@inline function collocgather!(z, x, e::CollocElem)
    Base.require_one_based_indexing(z, x)
    for k in 1:collocnv(e)
        z[k] = x[collocvar(e, k)]
    end
    return z
end

# ------------------------------------------------------------------ assembly
#
# Everything below is wiring. The mathematics is entirely in `colloc` and the
# two FSYS structs, and every derivative is AD over `CollocRowFun`.

"""
    _colloc_par(v, ref_rounding)

An interval endpoint as the element function should see it: the SIF's exact
value, or -- with `ref_rounding` -- rounded to the 11 significant decimal
digits SIFdecode's output format preserves. See the header, note 2.
"""
@inline _colloc_par(v::Real, ref_rounding::Bool) =
    ref_rounding ? round(v, sigdigits = 11) : v

"The element types these two SIFs use besides the external one."
const _COLLOC_PLAIN_ETYPES = ("SQR", "PROD")

"Bounds of one trivial-group constraint row (SIF kind, scale, RANGES)."
function _colloc_conbounds(g, ::Type{T}) where {T}
    s = g.scale
    s > 0.0 || error("non-positive scale $(s) on a trivial constraint group")
    r = g.range_ === nothing ? nothing : g.range_ / s
    if g.kind == 'E'
        r === nothing && return (zero(T), zero(T))
        return r >= 0 ? (zero(T), T(r)) : (T(r), zero(T))
    elseif g.kind == 'L'
        return (r === nothing ? T(-Inf) : T(-abs(r)), zero(T))
    elseif g.kind == 'G'
        return (zero(T), r === nothing ? T(Inf) : T(abs(r)))
    end
    error("unexpected constraint kind $(g.kind)")
end

"""
    _colloc_decode(name, sysfor, Val(NX), Val(NU); T, ref_param_rounding,
                   kwargs...) -> NamedTuple

Decode a `COLLOC`-family problem and sort its pieces into the buckets both
builders assemble from: the constraint bounds, the linear terms, constants,
SQR and PROD elements and objective terms of the non-external part, one
[`CollocElem`](@ref) per `Ktyp` element, and the FSYS `sysfor` returns for the
`COEFRI` parameters found.

Refuses anything it is not prepared to reproduce exactly -- a group type, an
unexpected element type, an objective element, internal variables, a QUADRATIC
section, a variable scale -- rather than quietly dropping it. `kwargs` go to
`ExaModelsSIF.sif_decode`, so a `-param` variant reaches the decode.
"""
function _colloc_decode(name::AbstractString, sysfor, ::Val{NX}, ::Val{NU};
                        T = Float64, ref_param_rounding::Bool = false,
                        kwargs...) where {NX,NU}
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer, kwargs...)
    tname = "Ktyp"
    haskey(P.external_types, tname) ||
        error("$name: expected a deferred $tname element type, found " *
              "$(collect(keys(P.external_types)))")
    length(P.external_types) == 1 ||
        error("$name: unexpected external types $(collect(keys(P.external_types)))")
    isempty(P.quad) ||
        error("$name: QUADRATIC section is not handled by colloc_model")
    all(==(1.0), P.varscale) ||
        error("$name: variable scales are not handled by colloc_model")

    et = P.eltypes[tname]
    isempty(et.ivars) ||
        error("$name: $tname declares internal variables $(et.ivars)")
    nv = 2NX + NU + 1
    length(et.evars) == nv ||
        error("$name: $tname has $(length(et.evars)) elemental variables, " *
              "expected $nv for NX=$NX, NU=$NU")

    nvar = length(P.varnames)
    connames = [gn for gn in P.grouporder if P.groups[gn].kind != 'N']
    rowof = Dict(gn => i for (i, gn) in enumerate(connames))
    ncon = length(connames)
    ncon > 0 || error("$name: no constraint rows")

    lcon = Vector{T}(undef, ncon)
    ucon = Vector{T}(undef, ncon)
    for (i, gn) in enumerate(connames)
        lcon[i], ucon[i] = _colloc_conbounds(P.groups[gn], T)
        lcon[i] <= -1e20 && (lcon[i] = T(-Inf))   # CUTEst's infinity threshold
        ucon[i] >= 1e20 && (ucon[i] = T(Inf))
    end

    # --- bucket the decode
    clin = NamedTuple{(:i, :j, :a),Tuple{Int,Int,T}}[]
    ccst = NamedTuple{(:i, :k),Tuple{Int,T}}[]
    csqr = NamedTuple{(:i, :j, :w),Tuple{Int,Int,T}}[]
    cprd = NamedTuple{(:i, :j, :k, :w),Tuple{Int,Int,Int,T}}[]
    olin = NamedTuple{(:j, :a),Tuple{Int,T}}[]
    objconst = zero(T)
    elems = CollocElem{NX,NU,T}[]
    coefri = Float64[]

    for gn in P.grouporder
        g = P.groups[gn]
        gt = g.gtype === nothing ? P.default_gtype : g.gtype
        (gt === nothing || gt == "TRIVIAL") ||
            error("$name: group $gn has group type $gt; colloc_model handles " *
                  "trivial groups only")
        iscon = g.kind != 'N'
        s = T(g.scale)
        if iscon
            i = rowof[gn]
            for (j, a) in g.lin
                push!(clin, (i = i, j = j, a = T(a) / s))
            end
            g.constant != 0.0 && push!(ccst, (i = i, k = -T(g.constant) / s))
        else
            for (j, a) in g.lin
                push!(olin, (j = j, a = T(a) / s))
            end
            objconst -= T(g.constant) / s
        end
        for (en, w0) in g.elems
            iscon || error("$name: objective group $gn carries element $en; " *
                           "colloc_model handles linear objectives only")
            i = rowof[gn]
            w = T(w0) / s
            e = P.elements[en]
            ten = isempty(e.etype) ? P.default_etype : e.etype
            if ten == tname
                push!(elems, CollocElem{NX,NU,T}(
                    ntuple(k -> e.varmap[Symbol("XI", k)], Val(NX)),
                    ntuple(k -> e.varmap[Symbol("XF", k)], Val(NX)),
                    ntuple(k -> e.varmap[Symbol("UV", k)], Val(NU)),
                    e.varmap[:QV1],
                    T(_colloc_par(e.parvals[:TN], ref_param_rounding)),
                    T(_colloc_par(e.parvals[:TN1], ref_param_rounding)),
                    Int(trunc(e.parvals[:ID])), i, w))
                haskey(e.parvals, :COEFRI) && push!(coefri, e.parvals[:COEFRI])
            elseif ten == "SQR"
                push!(csqr, (i = i, j = e.varmap[:X], w = w))
            elseif ten == "PROD"
                push!(cprd, (i = i, j = e.varmap[:X], k = e.varmap[:Y], w = w))
            else
                error("$name: element $en has type $ten; colloc_model handles " *
                      "$tname and $(_COLLOC_PLAIN_ETYPES)")
            end
        end
    end
    isempty(elems) && error("$name: no $tname elements found")
    for e in elems
        1 <= e.id <= NX ||
            error("$name: element ID $(e.id) outside 1:$NX")
    end

    # CAR2 passes the friction coefficient per element; it comes from one SIF
    # $-PARAMETER, so one FSYS serves the whole problem -- but check, rather
    # than read it off the first element and hope.
    isempty(coefri) || allequal(coefri) ||
        error("$name: COEFRI differs between elements " *
              "($(minimum(coefri)) .. $(maximum(coefri))); colloc_model " *
              "assumes one FSYS per problem")
    sys = sysfor(coefri)
    return (; P, nvar, ncon, lcon, ucon, clin, ccst, csqr, cprd, olin,
            objconst, elems, sys, nv)
end

"""
    _colloc_simd_core(T, backend, D) -> (core, x, con)

The non-external part of a decoded COLLOC-family problem as ordinary ExaModels
SIMD expressions: the linear objective, the group constants, the linear
constraint terms and the SQR/PROD elements, in SIF declaration order.

The constraint rows are declared by [`ExaModels.add_con`](@ref) before any of
them is filled, so row `i` is the `i`-th constraint group of the decode and a
later `add_eval` can fill rows that already exist.
"""
function _colloc_simd_core(::Type{T}, backend, D) where {T}
    (; P, nvar, ncon, lcon, ucon, clin, ccst, csqr, cprd, olin, objconst) = D
    # --- the SIMD part, filling rows by their SIF index
    c = ExaModels.ExaCore(T; backend)
    c, x = ExaModels.add_var(c, nvar; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c, con = ExaModels.add_con(c, ncon; lcon = lcon, ucon = ucon)
    isempty(ccst) ||
        ((c, _) = ExaModels.add_con!(c, con, Base.Generator(d -> d.i => d.k, ccst)))
    isempty(clin) ||
        ((c, _) = ExaModels.add_con!(c, con,
            Base.Generator(d -> d.i => d.a * x[d.j], clin)))
    isempty(csqr) ||
        ((c, _) = ExaModels.add_con!(c, con,
            Base.Generator(d -> d.i => d.w * x[d.j]^2, csqr)))
    isempty(cprd) ||
        ((c, _) = ExaModels.add_con!(c, con,
            Base.Generator(d -> d.i => d.w * x[d.j] * x[d.k], cprd)))
    isempty(olin) ||
        ((c, _) = ExaModels.add_obj(c, Base.Generator(d -> d.a * x[d.j], olin)))
    iszero(objconst) ||
        ((c, _) = ExaModels.add_obj(c, Base.Generator(d -> d.k, [(k = objconst,)])))
    return c, x, con
end

"""
    colloc_model(name, sysfor, Val(NX), Val(NU); T, backend,
                 element_hessian, ref_param_rounding, kwargs...) -> ExaModel

Build a `COLLOC`-family problem: the non-external part (linear objective,
SQR/PROD elements, linear terms, constants, group scales) as ordinary
ExaModels SIMD expressions in SIF declaration order, and the `Ktyp` elements
as one `ExaModels.add_eval` oracle over `colloc`.

`sysfor` receives the `COEFRI` element parameters found in the decode (empty
if the element type has none) and returns the FSYS to use, so a problem whose
right-hand side carries a SIF `\$-PARAMETER` reads it from the data instead of
hardcoding it.

`element_hessian = true`, the default, gives the TRUE Hessian by AD over the
ported value. It is the mathematically right one and it does NOT match the
reference, which has no COLLOC second derivatives at all.
`element_hessian = false` drops them to match -- what the SIF declares and
what SIFdecode, and so CUTEst, evaluates. See the header, note 1.

`ref_param_rounding = true` rounds the interval endpoints TN/TN1 to the 11
significant digits the reference's copies carry, which makes every row value
agree with the compiled Fortran bit for bit. The default `false` uses the
SIF's exact values; see the header for the measurement.

Refuses anything in the decode it is not prepared to reproduce exactly -- a
group type, an unexpected element type, an objective element, internal
variables, a QUADRATIC section, a variable scale -- rather than quietly
dropping it.
"""
function colloc_model(name::AbstractString, sysfor, ::Val{NX}, ::Val{NU};
                      T = Float64, backend = nothing, element_hessian::Bool = true,
                      ref_param_rounding::Bool = false,
                      kwargs...) where {NX,NU}
    # The oracle's callbacks run on the CPU, so a backend is refused rather
    # than ignored: accepting one would misreport the model as living on the
    # device. `colloc` itself is allocation- and branch-free and so GPU-safe.
    backend === nothing || error(
        "$name: colloc_model does not take a backend -- the COLLOC oracle's " *
        "callbacks run on the CPU. `colloc` itself is GPU-safe.")
    D = _colloc_decode(name, sysfor, Val(NX), Val(NU); T, ref_param_rounding,
                       kwargs...)
    (; elems, nv) = D
    sys = D.sys
    c, x, con = _colloc_simd_core(T, backend, D)

    # --- the external part, as an oracle over `colloc`
    nel = length(elems)
    zbuf = Vector{T}(undef, nv)
    gbuf = Vector{T}(undef, nv)
    hbuf = Matrix{T}(undef, nv, nv)
    # One config per model, shared by every element: CollocRowFun is a struct,
    # so all the elements' functions have the same type and so the same AD tag.
    gcfg = ForwardDiff.GradientConfig(CollocRowFun(sys, elems[1]), zbuf,
                                      ForwardDiff.Chunk{nv}())
    hcfg = element_hessian ?
        ForwardDiff.HessianConfig(CollocRowFun(sys, elems[1]), zbuf,
                                  ForwardDiff.Chunk{nv}()) : nothing

    f! = (res, xv) -> begin
        Base.require_one_based_indexing(res)
        fill!(res, zero(eltype(res)))
        for e in elems
            collocgather!(zbuf, xv, e)
            res[e.row] += e.w * CollocRowFun(sys, e)(zbuf)
        end
        nothing
    end
    jac! = (vals, xv) -> begin
        Base.require_one_based_indexing(vals)
        p = 0
        for e in elems
            collocgather!(zbuf, xv, e)
            ForwardDiff.gradient!(gbuf, CollocRowFun(sys, e), zbuf, gcfg)
            for k in 1:nv
                vals[p+k] = e.w * gbuf[k]
            end
            p += nv
        end
        nothing
    end
    hess! = (vals, xv, y) -> begin
        Base.require_one_based_indexing(vals)
        p = 0
        for e in elems
            collocgather!(zbuf, xv, e)
            ForwardDiff.hessian!(hbuf, CollocRowFun(sys, e), zbuf, hcfg)
            wy = e.w * y[e.row]
            for a in 1:nv, b in 1:a
                p += 1
                vals[p] = wy * hbuf[a, b]
            end
        end
        nothing
    end

    # Lower triangle of each element's dense nv x nv block. Pairs repeat
    # across elements (Q sits in every one); ExaModels sums coordinate
    # duplicates, and its hprod walks the entries additively.
    hstruct! = (rows, cols) -> begin
        for e in elems, a in 1:nv, b in 1:a
            i, j = collocvar(e, a), collocvar(e, b)
            push!(rows, max(i, j))
            push!(cols, min(i, j))
        end
    end
    c, _ = ExaModels.add_eval(c, (con,), (x,), f!;
        jac! = jac!,
        hess! = element_hessian ? hess! : nothing,
        nnzj = nv * nel,
        nnzh = element_hessian ? (nv * (nv + 1) ÷ 2) * nel : 0,
        jac_structure! = (rows, cols) -> begin
            for e in elems, k in 1:nv
                push!(rows, e.row)
                push!(cols, collocvar(e, k))
            end
        end,
        hess_structure! = element_hessian ? hstruct! : nothing,
        adapt = Val(false))

    return ExaModels.ExaModel(c)
end

"""
    car2_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

CAR2 -- minimum-time traverse of a point mass with friction -- with its
external `COLLOC` supplied by [`colloc`](@ref) and `CarODE` instead of by
compiled Fortran. `element_hessian` and `ref_param_rounding` are described
under [`colloc_model`](@ref); any other `kwargs` go to
`ExaModelsSIF.sif_decode`.
"""
car2_model(; T = Float64, backend = nothing, kwargs...) =
    colloc_model("CAR2", cf -> CarODE(T(only(unique(cf)))), Val(4), Val(2);
                 T, backend, kwargs...)

"""
    orbit2_model(; T = Float64, backend = nothing, kwargs...) -> ExaModel

ORBIT2 -- minimum-time low-thrust orbit transfer -- with its external
`COLLOC` supplied by [`colloc`](@ref) and [`orbit2_ode`](@ref) instead of by
compiled Fortran. `element_hessian` and `ref_param_rounding` are described
under [`colloc_model`](@ref); any other `kwargs` go to
`ExaModelsSIF.sif_decode`.
"""
orbit2_model(; T = Float64, backend = nothing, kwargs...) =
    colloc_model("ORBIT2", _ -> orbit2_ode(T), Val(6), Val(3);
                 T, backend, kwargs...)
