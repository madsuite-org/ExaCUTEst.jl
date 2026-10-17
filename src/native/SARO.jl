# SARO / SAROMM: a generic Julia port of their external Fortran.
#
# WHAT THE FORTRAN COMPUTES. Both SIF files append the SAME Fortran (the two
# appended blocks are byte-identical) and both declare the element types
# FTYPE and GTYPE with the single value routine `SAROFN(RK, RIDX, SYSEQ,
# X1..X9, U1, U2)`. `SAROFN` is a thin dispatcher: it packs its eleven
# elemental variables into X(1:9) and U(1:2), calls
#
#   SRFG(K, X, U, F, G)      K = RK
#     KK = 2*(K-1) + 1;  DT = 12
#     P(1) = U(1);  SRK4(KK,   DT, X,  P, FH, W);  G(1) = W(1)
#     P(1) = U(2);  SRK4(KK+1, DT, FH, P, F,  W);  G(2) = W(1)
#
#   SRK4(KK, DT, X, P, F, W) -- classical RK4 over one 12 h sample period,
#     four calls to SREQ; W is returned from the FIRST call only, i.e. it is
#     the algebraic output evaluated at the step's INPUT state.
#
#   SREQ(KK, X, P, DF, W) -- the DYMOLA-generated ODE right-hand side of the
#     solar plant: six store-layer temperatures X(1:6), three accumulated heat
#     quantities X(7:9), one control P(1) (mixing-valve position), driven by
#     day-indexed weather/load data held in COMMON /SARDAT/. W(1) is the
#     supplementary heat power.
#
# and returns F(RIDX) for FTYPE (SYSEQ = .TRUE.) or G(RIDX) for GTYPE. So one
# SIF stage k owns eleven element values -- F(1..9) and G(1..2) -- all of them
# functions of the same eleven variables X(1..9,k), U(1,k), U(2,k), and all of
# them produced by ONE `SRFG` call. `saro_stage_values` below is that call.
#
# SHAPE: GENERIC JULIA FUNCTION + VectorNonlinearOracle (not folded, not an
# expression). `SAROFN` depends on the variables, so folding is out. And the
# chain is deep: `SRFG` runs RK4 twice, i.e. EIGHT `SREQ` evaluations of ~60
# floating-point operations each, ~500 operations per element. An ExaModels
# expression nests one type parameter per operation, which makes a chain this
# long impractical to compile -- `src/native/BA.jl` records the same wall at
# ~30 operations. So this must stay a plain type-generic Julia function, with
# the derivatives coming from ForwardDiff over it. `SAROGN`, the
# ADIFOR-generated gradient routine the SIF also lists, is NOT ported.
#
# THE REFERENCE HAS NO SECOND DERIVATIVES. The SIF's `INDIVIDUALS` block gives
# `F` (value) and `A`/`G` (gradient, through `SAROGN`) but no `H` section, and
# the classification string `LOR1-MN-...` ends in `1`: first derivatives only.
# CUTEst therefore returns an identically ZERO Hessian for both problems --
# measured, `hess_coord` is all zeros and `hprod` is the zero vector for random
# multipliers at two points. This port's Hessian comes from AD and is the true
# one, so it does NOT agree with the reference, and it should not: it is the
# reference that is missing. What it can be checked against is a central
# difference of CUTEst's own Jacobian, which uses only derivatives the SIF does
# supply: the two agree to 6e-11 .. 4e-10, the accuracy a central difference can
# deliver.
#
# ONE ORACLE FOR ALL ROWS. Each row is (one element) + (one or two linear
# terms): F(i,k) is `FE(i,k) - X(i,k+1) = 0`, G(i,k) is `GE(i,k) - S(i,k) = 0`,
# and SAROMM adds purely linear rows (`FPSM(k)`: PSM(k) - PSM(k+1) = 0, and
# `GPSM(i,k)`: PSM(k) - S(i,k) >= 0). All of them go into a single
# `VectorNonlinearOracle` in SIF group order, so our row i is exactly
# `sif_constraint_names(...)[i]` and nothing has to be interleaved. The
# objective is linear in both problems (SARO: X(7,KMAX); SAROMM: PSM(KMAX)) and
# stays a normal ExaModels expression.
#
# THE PRECISION TRAP -- and it is load-bearing here. `SREQ` has no
# `implicit none`, and the eleven per-day data scalars are never declared:
#
#       DOUBLE PRECISION TAS(KMAX), TRSHS(KMAX), PSHS(KMAX)
#       ...
#       TA = TAS(K)
#       TRSH = TRSHS(K)
#
# Under Fortran's implicit typing a name starting with T, P, H or O is default
# REAL, i.e. SINGLE precision, so TA, TRSH, PSH, H0..H3 and OP0..OP3 are each
# rounded to Float32 on assignment even though the COMMON arrays they come from
# are double. ADIFOR saw the same thing and emitted `DBLE(TRSH)` / `DBLE(PSH)`
# throughout the generated derivative code, which confirms the reading. The
# tables are therefore stored Float32-rounded (`SAROTables`), and ignoring this
# moves the element value by ~3e-7 relative (measured at stages 1, 97, 200 and
# 365) -- not a rounding artifact, a different function. With the rounding in
# place, `cons` matches the Fortran to the last bit.
#
# DEAD BRANCH. `SREQ` tests `IF (EXTU)` where `EXTU` is a PARAMETER set to
# `.TRUE.`, so V(13) = P(1) unconditionally and the `ELSE` arm (a logistic /
# hyperbolic valve law in X(1)) is unreachable. Only the live arm is ported,
# which is also what keeps the function branch-free on values.
#
# WHY `saro_rhs` IS `@noinline`. `saro_stage_values` calls it eight times, and
# a stage's Hessian block needs it under twice-nested 11-partial duals, where
# every scalar becomes 144 Float64s. Measured both ways:
#
#               hess compile   hess per stage   365 stages
#   @inline        267.6 s         819 us        0.054 s
#   @noinline       10.5 s         254 us        0.030 s
#
# Inlined, the eight copies make one ~72000-value function that LLVM spends
# four and a half minutes optimizing -- and the result is SLOWER, because it
# spills. With the call left standing, one copy is compiled. (The Jacobian, a
# single sweep of 11-partial duals, compiles in 0.6 s and runs in 16 us per
# stage.) `@noinline` is a code-size decision only: the function is still
# allocation-free and still callable from a GPU kernel.
#
# LITERALS ARE KEPT AS WRITTEN. Constants such as 1161.11111111111,
# 69666.6666666667, 0.000277777777777778 and 9.41985645933014e-06 are
# truncated decimals of 10450/9, 627000/9, 1/3600 and 1/106158.9...; replacing
# them with the exact rationals they approximate shifts results at the 1e-15
# level. The Float64 literal is the reference, so `T(literal)` is what the port
# uses (in BigFloat that widens the Float64 value, which is the Fortran's own
# constant). Operation order inside each assignment is also kept, so Float64
# evaluation reproduces the Fortran bit pattern rather than merely its value.

# ---------------------------------------------------------------------------
# The per-day driving data (COMMON /SARDAT/, filled by BLOCK DATA SARCOM)

"""
    SAROTables

The eleven day-indexed driving signals of the S\"ar\"o model, one entry per
communication point (366 of them): ambient temperature `ta`, space-heating
return temperature `trsh`, heating power `psh`, the cubic coefficients
`h0..h3` of collector heat and `op0..op3` of collector mass flow.

Values are stored Float32-rounded, because `SREQ` assigns them to implicitly
typed (single precision) scalars -- see the header.
"""
struct SAROTables
    ta::Vector{Float64}
    trsh::Vector{Float64}
    psh::Vector{Float64}
    h0::Vector{Float64}
    h1::Vector{Float64}
    h2::Vector{Float64}
    h3::Vector{Float64}
    op0::Vector{Float64}
    op1::Vector{Float64}
    op2::Vector{Float64}
    op3::Vector{Float64}
end

# TA/TRSH/... are stored in COMMON as four consecutive chunks of 108, 108, 108
# and 42 entries; the names in BLOCK DATA are the chunks, the names in SREQ are
# the whole 366-long arrays.
const _SARO_CHUNKS = ("1", "2", "3", "4")

"Parse every `DATA name/.../` initializer out of `BLOCK DATA SARCOM`."
function _saro_data_statements(text::AbstractString)
    i = findfirst("BLOCK DATA SARCOM", text)
    i === nothing && error("SARO: no BLOCK DATA SARCOM in the SIF file")
    body = text[first(i):end]
    j = findfirst(r"\n {6}END\b", body)
    j === nothing && error("SARO: BLOCK DATA SARCOM is not terminated")
    body = body[1:first(j)]
    # drop comment lines, then drop Fortran columns 1-6 (label + continuation
    # mark) so the statements join into one stream
    buf = IOBuffer()
    for ln in eachsplit(body, '\n')
        (isempty(ln) || ln[1] == 'C' || ln[1] == 'c' || ln[1] == '*') && continue
        print(buf, length(ln) > 6 ? ln[7:end] : "")
    end
    joined = String(take!(buf))
    out = Dict{String,Vector{Float64}}()
    for m in eachmatch(r"DATA\s+([A-Z0-9]+)/([^/]+)/", joined)
        vals = Float64[]
        for tok in eachsplit(m.captures[2], ',')
            s = strip(tok)
            isempty(s) && continue
            push!(vals, parse(Float64, replace(s, 'D' => 'e', 'd' => 'e')))
        end
        out[m.captures[1]] = vals
    end
    return out
end

"""
    saro_tables(path) -> SAROTables

Read the `BLOCK DATA SARCOM` driving data appended to a SARO/SAROMM SIF file.

The four 108/108/108/42-long COMMON chunks of each signal are concatenated into
one 366-long vector and rounded through Float32, matching the implicit single
precision of the scalars `SREQ` copies them into.
"""
function saro_tables(path::AbstractString)
    d = _saro_data_statements(read(path, String))
    get1 = function (stem)
        v = Float64[]
        for ch in _SARO_CHUNKS
            k = stem * ch
            haskey(d, k) || error("SARO: BLOCK DATA has no array $k")
            append!(v, d[k])
        end
        length(v) == 366 || error("SARO: $stem has $(length(v)) entries, expected 366")
        # the Fortran rounds each value to single precision on assignment
        return Float64[Float64(Float32(z)) for z in v]
    end
    return SAROTables(get1("TA"), get1("TRSH"), get1("PSH"),
                      get1("H0"), get1("H1"), get1("H2"), get1("H3"),
                      get1("OP0"), get1("OP1"), get1("OP2"), get1("OP3"))
end

const _SARO_TABLE_CACHE = Dict{String,SAROTables}()
_saro_tables_cached(path::AbstractString) =
    get!(() -> saro_tables(path), _SARO_TABLE_CACHE, abspath(path))

"""
    saro_stage_data(tab, k, T) -> (da, db)

The two 11-tuples of driving data `SREQ` sees during SIF stage `k`, in the
order `(ta, trsh, psh, h0, h1, h2, h3, op0, op1, op2, op3)`.

`SRFG` splits a stage into two 12 h sample periods, `KK = 2(k-1)+1` and
`KK+1`, which both map to day `K = (KK-1) >> 1 + 1 = k`. The collector terms
are applied in the FIRST period only, at twice their daily mean (`H0 =
2*H0S(K)` when `2*(K-1) == KK-1`, zero otherwise), so `da` carries them and
`db` zeros them. Doubling is exact in binary, so it commutes with the stored
Float32 rounding.
"""
function saro_stage_data(tab::SAROTables, k::Int, ::Type{T}) where {T}
    ta, trsh, psh = T(tab.ta[k]), T(tab.trsh[k]), T(tab.psh[k])
    two = T(2)
    da = (ta, trsh, psh,
          two * T(tab.h0[k]), two * T(tab.h1[k]), two * T(tab.h2[k]), two * T(tab.h3[k]),
          two * T(tab.op0[k]), two * T(tab.op1[k]), two * T(tab.op2[k]), two * T(tab.op3[k]))
    z = zero(T)
    db = (ta, trsh, psh, z, z, z, z, z, z, z, z)
    return da, db
end

# ---------------------------------------------------------------------------
# The port proper: SREQ -> SRK4 -> SRFG, as generic Julia functions

"""
    saro_rhs(d, X, p) -> (DF, w)

`SREQ`: the right-hand side of the continuous-time solar-plant ODE, plus the
algebraic supplementary heat power `W(1)`.

`d` is the 11-tuple from [`saro_stage_data`](@ref), `X` the nine states and `p`
the mixing-valve control. Generic in the element type, allocation-free, and
branch-free on values (the `EXTU` branch is dead -- see the header). `d` may be
in the working precision while `X` is a `ForwardDiff.Dual` over it; the
literals follow `d`'s type, which is what keeps a Float32 model in Float32.

Marked `@noinline` on purpose: it is called eight times per element and under
twice-nested duals, so one compiled copy instead of eight is the difference
between a 10 s and a multi-minute Hessian compile. See the header.
"""
@noinline function saro_rhs(d::NTuple{11,S}, X::NTuple{9,T}, p::T) where {S,T}
    ta, trsh, psh, h0, h1, h2, h3, op0, op1, op2, op3 = d
    # X(7:9) are pure accumulators: they appear in DF(7:9) nowhere, only in the
    # RK4 update, so the right-hand side reads X(1:6) only.
    x1, x2, x3, x4, x5, x6 = X[1], X[2], X[3], X[4], X[5], X[6]

    v12 = S(0.001) * psh
    v13 = p                                           # EXTU = .TRUE.
    v6 = S(1000.0) * v13 * v12 /
         (S(69666.6666666667) - S(1161.11111111111) * trsh)
    v11 = S(1.16111111111111) * (x1 - trsh) * v6
    v10 = v12 - v11
    df7 = S(0.001) * v10

    v19 = op0 + (op1 + (op2 + op3 * x6) * x6) * x6
    v14 = S(740.0) * v19
    v16 = S(0.000277777777777778) * v14
    v43 = S(3600.0) * v16                             # == V61 == V78 == V95 == V112 == V129
    v18 = h0 + (h1 + (h2 + h3 * x6) * x6) * x6
    v46 = x6 + v18 / (S(1.16111111111111e-27) + S(1161.11111111111) * v19)
    v8 = S(0.000277777777777778) * v6
    v48 = S(3600.0) * v8                              # == V65 == V82 == V99 == V116 == V133

    c = S(1161.11111111111)
    v166 = S(47.5428571428572) * (x1 - x2)
    v157 = S(10.1587301587302) * (x1 - ta)
    v40 = -(v166 + v157)
    v138 = S(3.76619850066144) * (x1 - ta)
    df1 = S(9.41985645933014e-06) *
          (c * v43 * (v46 - x1) + c * v48 * (x2 - x1) + v40 - v138)

    v170 = S(39.6190476190476) * (x2 - x3)
    v58 = v166 - v170
    v142 = S(4.0) + S(0.5) * ta
    v141 = S(5.64929775099216) * (x2 - v142)
    df2 = S(6.27990430622009e-06) *
          (c * v43 * (x1 - x2) + c * v48 * (x3 - x2) + v58 - v141)

    v175 = S(39.6190476190476) * (x3 - x4)
    v75 = v170 - v175
    v145 = S(5.64929775099216) * x3 - S(45.1943820079373)
    df3 = S(6.27990430622009e-06) *
          (c * v43 * (x2 - x3) + c * v48 * (x4 - x3) + v75 - v145)

    v180 = S(47.5428571428572) * (x4 - x5)
    v92 = v175 - v180
    v148 = S(5.64929775099216) * x4 - S(45.1943820079373)
    df4 = S(6.27990430622009e-06) *
          (c * v43 * (x3 - x4) + c * v48 * (x5 - x4) + v92 - v148)

    v185 = S(79.2380952380953) * (x5 - x6)
    v109 = v180 - v185
    v151 = S(3.76619850066144) * x5 - S(30.1295880052915)
    df5 = S(9.41985645933014e-06) *
          (c * v43 * (x4 - x5) + c * v48 * (x6 - x5) + v109 - v151)

    v161 = S(73.1428571428572) * x6 - S(585.142857142857)
    v126 = v185 - v161
    v154 = S(1.88309925033072) * x6 - S(15.0647940026458)
    df6 = S(1.88397129186603e-05) *
          (c * v43 * (x5 - x6) + c * v48 * (trsh - x6) + v126 - v154)

    v196 = S(4180.0) * (x1 - trsh) * v8
    df8 = S(0.001) * v196
    v197 = S(4180.0) * (v46 - x6) * v16
    df9 = S(0.001) * v197

    return (df1, df2, df3, df4, df5, df6, df7, df8, df9), v10
end

"""
    saro_rk4(d, X, p) -> (F, w)

`SRK4`: one classical RK4 step of length `DT = 12` over [`saro_rhs`](@ref).

`w` is the algebraic output of the FIRST stage evaluation, i.e. at `X` itself
-- the Fortran passes `W` to the first `SREQ` call and a scratch `W1` to the
other three, so the later values are discarded. `DT/6`, `DT/3`, `DT/2` and
`DT` are 2, 4, 6 and 12 exactly, so no rounding is introduced by the weights.
"""
@inline function saro_rk4(d::NTuple{11,S}, X::NTuple{9,T}, p::T) where {S,T}
    dt = S(12.0)
    w6, w3, hh = dt / S(6.0), dt / S(3.0), dt / S(2.0)

    k1, w = saro_rhs(d, X, p)
    f1 = ntuple(i -> X[i] + w6 * k1[i], Val(9))
    y2 = ntuple(i -> X[i] + hh * k1[i], Val(9))

    k2, _ = saro_rhs(d, y2, p)
    f2 = ntuple(i -> f1[i] + w3 * k2[i], Val(9))
    y3 = ntuple(i -> X[i] + hh * k2[i], Val(9))

    k3, _ = saro_rhs(d, y3, p)
    f3 = ntuple(i -> f2[i] + w3 * k3[i], Val(9))
    y4 = ntuple(i -> X[i] + dt * k3[i], Val(9))

    k4, _ = saro_rhs(d, y4, p)
    f4 = ntuple(i -> f3[i] + w6 * k4[i], Val(9))
    return f4, w
end

"""
    saro_stage_values(da, db, v) -> NTuple{11}

`SRFG`: every element value SIF stage `k` owns, from the stage's eleven
variables `v = (X1..X9, U1, U2)`.

Slots 1..9 are `F(1..9)` -- the state at the end of the stage, two 12 h RK4
sample periods on from `X` -- and slots 10, 11 are `G(1)`, `G(2)`, the
supplementary heat power at the start of each sample period. `da`/`db` come
from [`saro_stage_data`](@ref).

This is the whole of `SAROFN`: it returns the value for every `(RK, RIDX,
SYSEQ)` the SIF asks about, at the cost of ONE evaluation instead of eleven.
"""
@inline function saro_stage_values(da::NTuple{11,S}, db::NTuple{11,S},
                                   v::NTuple{11,T}) where {S,T}
    X = (v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9])
    u1, u2 = v[10], v[11]
    fh, g1 = saro_rk4(da, X, u1)
    f, g2 = saro_rk4(db, fh, u2)
    return (f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], g1, g2)
end

# ---------------------------------------------------------------------------
# Derivatives, by AD over the one function above

"The 11x11 Jacobian of [`saro_stage_values`](@ref), one forward sweep."
@inline function saro_stage_jac(da::NTuple{11,T}, db::NTuple{11,T},
                                v::NTuple{11,T}) where {T}
    D = ForwardDiff.Dual{:SARO,T,11}
    vd = ntuple(i -> D(v[i],
                       ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(T), zero(T)),
                                                   Val(11)))),
                Val(11))
    out = saro_stage_values(da, db, vd)
    return ntuple(s -> ForwardDiff.partials(out[s]), Val(11))
end

"""
    saro_stage_hess(da, db, v, w) -> Matrix

The 11x11 Hessian of `sum_s w[s] * saro_stage_values(...)[s]`, i.e. a stage's
block of the Lagrangian Hessian with `w[s]` the multiplier of the row that
holds slot `s`.
"""
function saro_stage_hess(da::NTuple{11,T}, db::NTuple{11,T},
                         v::NTuple{11,T}, w::NTuple{11,T}) where {T}
    f = function (z)
        out = saro_stage_values(da, db, ntuple(i -> z[i], Val(11)))
        s = w[1] * out[1]
        for i in 2:11
            s += w[i] * out[i]
        end
        return s
    end
    return ForwardDiff.hessian(f, collect(v))
end

# ---------------------------------------------------------------------------
# Assembly: reading the SIF structure and wiring the oracle

"The eleven elemental variables of FTYPE/GTYPE, in SIF declaration order."
const SARO_EVARS = (:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :U1, :U2)

"""
    SAROStage

One SIF stage: the day index `k`, where its eleven variables live, and which
constraint row (if any) consumes each of the eleven element slots. A zero in
`rows` means the SIF never asks for that slot.
"""
struct SAROStage
    k::Int
    idx::NTuple{11,Int}
    rows::NTuple{11,Int}
end

"""
    saro_stages(P) -> (Vector{SAROStage}, connames, objterms)

Group the deferred FTYPE/GTYPE elements of a `sif_decode(...; external =
:defer)` problem by their `RK` parameter.

Every such element must have weight 1 in its row and `RIDX` in 1..9 (FTYPE) or
1..2 (GTYPE); elements sharing an `RK` must agree on their variable map, which
is what makes one `SRFG` call serve the whole stage. Anything else means the
SIF structure changed and is an error rather than a silent approximation.
"""
function saro_stages(P)
    connames = [g for g in P.grouporder if P.groups[g].kind != 'N']
    rowof = Dict(g => i for (i, g) in enumerate(connames))

    byk = Dict{Int,Tuple{NTuple{11,Int},Vector{Int}}}()
    order = Int[]
    for gn in connames
        g = P.groups[gn]
        length(g.elems) <= 1 ||
            error("SARO: row $gn holds $(length(g.elems)) elements; expected 0 or 1")
        isempty(g.elems) && continue
        en, wt = g.elems[1]
        wt == 1.0 || error("SARO: row $gn weights its element by $wt; expected 1")
        e = P.elements[en]
        et = isempty(e.etype) ? "" : e.etype
        base = et == "FTYPE" ? 0 : et == "GTYPE" ? 9 :
               error("SARO: element $en has unexpected type $et")
        ridx = Int(e.parvals[:RIDX])
        lim = base == 0 ? 9 : 2
        1 <= ridx <= lim || error("SARO: element $en has RIDX = $ridx, out of 1:$lim")
        k = Int(e.parvals[:RK])
        idx = ntuple(j -> e.varmap[SARO_EVARS[j]], Val(11))
        if haskey(byk, k)
            idx0, slots = byk[k]
            idx0 == idx || error("SARO: elements of stage $k disagree on their " *
                                 "variable map ($idx0 vs $idx)")
        else
            byk[k] = (idx, zeros(Int, 11))
            push!(order, k)
        end
        slots = byk[k][2]
        slot = base + ridx
        slots[slot] == 0 ||
            error("SARO: stage $k has two elements for slot $slot")
        slots[slot] = rowof[gn]
    end

    stages = [SAROStage(k, byk[k][1], ntuple(s -> byk[k][2][s], Val(11))) for k in order]

    # the objective: linear in both problems
    objterms = Tuple{Int,Float64}[]
    for gn in P.grouporder
        g = P.groups[gn]
        g.kind == 'N' || continue
        isempty(g.elems) ||
            error("SARO: objective group $gn has nonlinear elements")
        g.constant == 0.0 ||
            error("SARO: objective group $gn has constant $(g.constant)")
        g.scale == 1.0 || error("SARO: objective group $gn is scaled by $(g.scale)")
        for (j, a) in g.lin
            push!(objterms, (j, a))
        end
    end
    return stages, connames, objterms
end

"""
    saro_oracle_model(name; T = Float64, backend = nothing) -> ExaModel

Build SARO or SAROMM as an ExaModel whose constraints are the generic Julia
port of `SAROFN` rather than compiled Fortran.

All rows go into one `VectorNonlinearOracle` in SIF group order (so row `i` is
`sif_constraint_names(name)[i]`); each row's linear terms are added inside the
oracle, and the linear objective is a normal ExaModels expression. Derivatives
come from ForwardDiff over [`saro_stage_values`](@ref) -- the SIF's
ADIFOR-generated `SAROGN` is not used.

Needs no Fortran toolchain and no `sifdecoder`: the structure comes from
`sif_decode(...; external = :defer)` and the driving data from the
`BLOCK DATA SARCOM` appended to the same file.

`T != Float64` builds, and `cons` evaluates (agreeing with the Float64 model
to 5e-6 in Float32), but `jac` and `hess` do NOT, for two independent reasons
that are worth separating because fixing either alone leaves the other.

ExaModels allocates the oracle cache's callback buffers as untyped `zeros(n)`,
so the callbacks here see `E = eltype(vals) === Float64` and convert the
variables to `E` while `stagedata` stays in `T`. `saro_rk4` ties the control's
type to the state's, so the mixed pair has no method:
`saro_stage_jac(::NTuple{11,Float32}, ::NTuple{11,Float32}, ::NTuple{11,Float64})`.
Typing those buffers upstream removes it; relaxing the control's type parameter
here would harden the port against it.

In Float32 the Jacobian and Hessian are then NaN at every stage. The second
RK4 call zeroes the collector terms, so `saro_rhs` forms
`x6 + v18/(1.16111111111111e-27 + 1161.11 v19)` with `v18 = v19 = 0`, and
ForwardDiff's quotient rule squares the denominator: `Float32(1.161e-27)^2`
underflows to zero and gives `0/0`, where Float64 gives a normal 1.35e-54. The
regulariser is the Fortran's constant and stays; a Float32-safe port needs the
division restructured so it is never squared.

[`saro_stage_values`](@ref) itself is generic and evaluates in Float32 and
BigFloat whenever the data and the values share a type.
"""
function saro_oracle_model(name::AbstractString; T = Float64, backend = nothing)
    # Refuse a backend rather than accept and ignore it: that would report the
    # model as living on the device when its callbacks run on the host.
    backend === nothing || error(
        "saro_oracle_model does not take a backend: the port's callbacks run " *
        "on the CPU. saro_stage_values is GPU-safe, so a device path is a " *
        "matter of calling it from a kernel.")

    path = sif_problem_path(name)
    P = ExaModelsSIF.sif_decode(path; external = :defer)
    for (tname, fn) in P.external_types
        (tname in ("FTYPE", "GTYPE") && fn === :SAROFN) || error(
            "$name: expected deferred FTYPE/GTYPE calling SAROFN, found " *
            "$tname => $fn")
    end
    all(==(1.0), P.varscale) || error("$name: variable scaling is not supported here")

    tab = _saro_tables_cached(path)
    stages, connames, objterms = saro_stages(P)
    isempty(stages) && error("$name: no FTYPE/GTYPE elements found")

    n = length(P.varnames)
    ncon = length(connames)

    # --- row bounds and linear parts, straight from the decode
    lcon = zeros(T, ncon)
    ucon = zeros(T, ncon)
    rconst = zeros(T, ncon)
    lrow = Int[]
    lcol = Int[]
    lval = T[]
    for (i, gn) in enumerate(connames)
        g = P.groups[gn]
        g.scale == 1.0 || error("$name: row $gn is scaled by $(g.scale)")
        g.range_ === nothing || error("$name: row $gn has a range")
        # CUTEst (and ExaModelsSIF's builder) fold a group's CONSTANT into the
        # row VALUE and leave the bound at zero, so `cons(x)` is the residual.
        rconst[i] = -T(g.constant)
        if g.kind == 'E'
            lcon[i] = zero(T); ucon[i] = zero(T)
        elseif g.kind == 'G'
            lcon[i] = zero(T); ucon[i] = T(Inf)
        elseif g.kind == 'L'
            lcon[i] = T(-Inf); ucon[i] = zero(T)
        else
            error("$name: row $gn has unexpected kind $(g.kind)")
        end
        for (j, a) in g.lin
            push!(lrow, i); push!(lcol, j); push!(lval, T(a))
        end
    end
    nlin = length(lval)

    # --- stage data, materialized once
    stagedata = [saro_stage_data(tab, s.k, T) for s in stages]
    nst = length(stages)

    # --- Jacobian sparsity: the linear entries, then 11 per element row.
    # Coordinate duplicates are summed by ExaModels, so overlap would be
    # harmless; in these two problems there is none (a row's linear variable
    # is X(i,k+1) or S(i,k), never one of the stage's own eleven).
    nj = nlin + 11 * sum(count(!=(0), s.rows) for s in stages)
    jr = Vector{Int}(undef, nj)
    jc = Vector{Int}(undef, nj)
    copyto!(view(jr, 1:nlin), lrow)
    copyto!(view(jc, 1:nlin), lcol)
    let p = nlin
        for s in stages, slot in 1:11
            r = s.rows[slot]
            r == 0 && continue
            for a in 1:11
                p += 1
                jr[p] = r
                jc[p] = s.idx[a]
            end
        end
    end

    # --- Hessian sparsity: the lower triangle of each stage's 11x11 block.
    hr = Vector{Int}(undef, 66 * nst)
    hc = Vector{Int}(undef, 66 * nst)
    let p = 0
        for s in stages, a in 1:11, b in 1:a
            p += 1
            i, j = s.idx[a], s.idx[b]
            hr[p], hc[p] = max(i, j), min(i, j)
        end
    end

    f! = function (cv, xv)
        Base.require_one_based_indexing(cv, xv)
        E = eltype(cv)
        copyto!(cv, rconst)
        for p in eachindex(lrow, lcol, lval)
            cv[lrow[p]] += E(lval[p]) * xv[lcol[p]]
        end
        for (si, s) in enumerate(stages)
            da, db = stagedata[si]
            out = saro_stage_values(da, db,
                                    ntuple(a -> E(xv[s.idx[a]]), Val(11)))
            for slot in 1:11
                r = s.rows[slot]
                r == 0 && continue
                cv[r] += out[slot]
            end
        end
        return cv
    end

    jac! = function (vals, xv)
        Base.require_one_based_indexing(vals, xv)
        E = eltype(vals)
        for p in eachindex(lval)
            vals[p] = E(lval[p])
        end
        p = nlin
        for (si, s) in enumerate(stages)
            da, db = stagedata[si]
            J = saro_stage_jac(da, db, ntuple(a -> E(xv[s.idx[a]]), Val(11)))
            for slot in 1:11
                s.rows[slot] == 0 && continue
                g = J[slot]
                for a in 1:11
                    p += 1
                    vals[p] = g[a]
                end
            end
        end
        return vals
    end

    hess! = function (vals, xv, y)
        Base.require_one_based_indexing(vals, xv, y)
        E = eltype(vals)
        p = 0
        for (si, s) in enumerate(stages)
            da, db = stagedata[si]
            w = ntuple(slot -> s.rows[slot] == 0 ? zero(E) : E(y[s.rows[slot]]), Val(11))
            if all(iszero, w)
                for _ in 1:66
                    p += 1
                    vals[p] = zero(E)
                end
                continue
            end
            H = saro_stage_hess(da, db, ntuple(a -> E(xv[s.idx[a]]), Val(11)), w)
            for a in 1:11, b in 1:a
                p += 1
                vals[p] = H[a, b]
            end
        end
        return vals
    end

    c = ExaModels.ExaCore(T)
    c, x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    if !isempty(objterms)
        data = [(j = j, a = T(a)) for (j, a) in objterms]
        c, _ = ExaModels.add_obj(c, Base.Generator(d -> d.a * x[d.j], data))
    end
    vor = ExaModels.VectorNonlinearOracle(;
        nvar = n, ncon = ncon,
        jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
        lcon = lcon, ucon = ucon,
        f! = f!, jac! = jac!, hess! = hess!,
        adapt = Val(true))
    c = ExaModels.constraint(c, vor)
    return ExaModels.ExaModel(c)
end

"""
    saro_model(; T = Float64, backend = nothing) -> ExaModel

SARO: minimize the supplementary heat energy of the S\"ar\"o seasonal-store
solar plant over 365 stages, with the heat load bounded by `PSMAX` through the
slack variables' upper bound. See [`saro_oracle_model`](@ref).
"""
saro_model(; kwargs...) = saro_oracle_model("SARO"; kwargs...)

"""
    saromm_model(; T = Float64, backend = nothing) -> ExaModel

SAROMM: the min-max reformulation of [`saro_model`](@ref) -- minimize the
maximal supplementary heat load instead of bounding it. Same Fortran, same
elements; the extra `PSM` variables and the `FPSM`/`GPSM` rows are linear.
"""
saromm_model(; kwargs...) = saro_oracle_model("SAROMM"; kwargs...)
