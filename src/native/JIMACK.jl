# JIMACK: a generic Julia port of the external Fortran appended to JIMACK.SIF.
#
# WHAT THE FORTRAN COMPUTES. The box [-1,1]x[-1,1]x[0,DELTA] is meshed with
# N x N x M trilinear hexahedra (N = 12, M = 6, so 864 elements and
# 3*(N+1)^2*(M+1) = 3549 variables: the deformed position (AX,AY,AZ) of every
# grid node). One SIF element `E(I,J,K)` of type `INTEGRAL` owns the 24 nodal
# values of its 8 vertices and returns the Gauss-quadrature approximation of
# the stored energy of an isotropic nonlinear elastic material over that cell.
# The appended Fortran is three routines, declared in the SIF as `F` / `IEINT`:
#
#   IEINT(I,J,K, EPS,DELTA, DELX,DELY,DELZ, DX,DY,DZ, 24 nodal values, ...)
#     builds the cell corners from (I,J,K), walks 8 quadrature points, calls
#     IPHI at each, and returns INTF = sum of the 8 pointwise energies P(L1).
#     (It also fills the gradient/Hessian output arrays; NOT ported -- see
#     below.)
#
#   IPHI(X,Y,Z, ...) evaluates the pointwise energy at one quadrature point:
#     PSI gives the gradient G(vertex, dir) of each of the 8 trilinear shape
#     functions; grad u = sum over vertices of (nodal value) * G; the
#     right Cauchy-Green-like matrix is F = (grad u)(grad u)^T; its three
#     principal invariants INV1, INV2, INV3 go into
#
#       V   = 1 - EPS*Z/DELTA          (a depth-dependent stiffness taper)
#       P   = 0.65*INV1 + (0.45/V^2)*INV2 - 3.1*V^2*( log(INV3) - 3*log(V^2) )
#
#   PSI(X,Y,Z, ..., DX,DY,DZ, SXL,SYL,SZL, GX,GY,GZ) returns 1 and, through
#     GX/GY/GZ, the shape-function gradient of one vertex. Only the gradients
#     are used; the returned value is discarded.
#
# Both `IEINT` and `IPHI` are `INTEGER FUNCTION`s, and `IEINT` is declared `I
# IEINT` in the SIF's TEMPORARIES. That is not a precision trap: their return
# values are 1 and a dummy `IINT` that nothing reads. Every real result travels
# through the DOUBLE PRECISION argument `INTF`, which SIFdecode wires to the
# element value `F`. `PSI` is a genuine `DOUBLE PRECISION FUNCTION` returning a
# constant 1.0, also discarded.
#
# TWO FORTRAN QUIRKS THAT ARE PART OF THE DEFINITION. Both are reproduced here
# verbatim, because CUTEst's answer is what this port has to match.
#
#  1. The quadrature loop tests the WRONG index twice:
#
#         DO 180 K1 = 1, 2
#          IF ( K1 .EQ. 1 ) THEN  Z = ZMID - DELZ  ELSE  Z = ZMID + DELZ
#          DO 170 J1 = 1, 2
#           IF ( K1 .EQ. 1 ) THEN  Y = YMID - DELY  ELSE  Y = YMID + DELY
#           DO 160 I1 = 1, 2
#            IF ( K1 .EQ. 1 ) THEN  X = XMID - DELX  ELSE  X = XMID + DELX
#
#     `J1` and `I1` are never consulted, so there are only TWO distinct
#     quadrature points -- the cell centre shifted by (-DELX,-DELY,-DELZ) and
#     by (+DELX,+DELY,+DELZ) -- each visited four times. The element value is
#     therefore 4*P(minus) + 4*P(plus), accumulated in the Fortran's order.
#
#  2. `DELY` is set to `DX`, not to `HALFW*DY`:
#
#         A  DELX                W / RN        (= HALFW * DX, as intended)
#         A  DELY                DX            (a full cell width)
#         A  DELZ                HALFW * DELTA / RM
#
#     so the quadrature points sit OUTSIDE the cell in y. Harmless for this
#     port (the shape-function gradients are evaluated wherever they are
#     asked for) but it is why `dely != delx*dy/dx`.
#
# THE PRECISION FINDING -- and it is real, not cosmetic. The SIF's GLOBALS
# section writes three constants bare, with no `D` exponent, and SIFdecode
# copies the text verbatim into `ELFUN_JIMACK.f`:
#
#       DOUBLE PRECISION DELTA , EPS   , W     , HALFW , F
#       DELTA  = 0.1
#       EPS    = 0.01
#       W      = 0.577350269189
#
# A bare Fortran real literal is default REAL, i.e. SINGLE precision, and
# gfortran (CUTEst compiles with plain `-O3`, no `-fdefault-real-8`) rounds it
# to Float32 before widening it into the DOUBLE PRECISION variable. So the
# constants the Fortran actually uses are
#
#       DELTA = 0.10000000149011612
#       EPS   = 0.009999999776482582
#       W     = 0.5773502588272095
#
# and NOT 0.1, 0.01, 0.577350269189. The data DISCRIMINATES, decisively.
# Measured against `CUTEstModel("JIMACK")` at x0 and at a perturbed point
# (`test/native_verify.jl`), with the group-scale artifact below taken out:
#
#                       obj rel      grad rel (2-norm)
#     :single (Fortran)  0 (bitwise)  5.3e-16 / 6.2e-15
#     :double (naive)    1.5e-8       3.1e-8  / 2.2e-6
#
# `:single` is EXACT -- the objective comes back bit for bit -- and `:double`
# misses by 1e-8, eight orders of magnitude out and nowhere near roundoff, i.e.
# a different function. `fortran_literals = :double` is kept only so the two
# readings can be compared; `:single` is the default and is the Fortran.
# Note `DELTA` ALSO appears in the SIF DATA sections, where it is a proper
# double 0.1 and sets the objective's group scale to 2*M*N^2/DELTA = 17280.
# Both readings use that same scale; only the Fortran's copy is rounded.
# Everything else in the appended Fortran carries an explicit double exponent
# (`4.5D-1`, `3.1D+0`, `6.5D-1`, `5.0D-1`, `2.0D+0 / RN`), and `IEINT`, `IPHI`
# and `PSI` declare every variable they use, so there is no implicit-typing
# trap inside the three routines -- only in the SIF-generated preamble.
#
# A SECOND, SEPARATE PRECISION LOSS, AND IT IS CUTEst'S. SIFdecode writes the
# reciprocal group scale into OUTSDIF.d in an 11-significant-digit `E` format:
#
#     $ grep D-05 OUTSDIF_JIMACK.d   ->   5.7870370370D-05
#
# while 1/17280 = 5.787037037037037e-5. CUTEst reads that truncated number back
# and multiplies the group value by it, so every CUTEst quantity for this
# problem -- obj, grad, Hessian alike -- is uniformly (1 - 6.400e-12) times the
# exactly scaled one. This model divides by 17280 instead, so a raw comparison
# against CUTEst shows a FLAT 6.400e-12 relative difference at every point and
# in every quantity. That is CUTEst's text round-trip, not this port: multiply
# by `5.7870370370e-5 * 17280` and the objective agrees bit for bit. (The same
# 11-digit format is why CUTEst's x0 differs from the decode at ~3e-12; the
# verification script evaluates both models at CUTEst's x0, so that cancels.)
# Do not "fix" this by hard-coding the truncated constant.
#
# SHAPE: a generic Julia function behind a `ScalarNonlinearOracle`.
# Not folded: the energy depends on the element's 24 VARIABLES, so it is not a
# constant per element the way VANDANIUM's B-spline is. Not an ExaModels
# expression either: one element value is ~160 floating-point operations (72
# multiply-adds for grad u, the 6 entries of F, the 3 invariants, 2 logs,
# twice over for the two quadrature points), and ExaModels nests a type
# parameter per operation -- a ~30-deep chain already failed to compile in an
# hour on a 57-variable problem (see src/native/BA.jl). So the value is a
# plain, type-generic Julia function and ForwardDiff supplies the derivatives.
# The Fortran's hand-coded `INTGX/INTGY/INTGZ` gradient and `INTHXX..INTHZZ`
# Hessian arrays -- about 400 lines of IPHI, most of the file -- are
# deliberately NOT ported.
#
# PER-ELEMENT AD, THEN SCATTER. The objective is a sum over 864 elements, so
# the AD unit is ONE element: 24 variables in, one scalar out. The gradient is
# one 24-partial forward sweep per element and the Hessian-vector product is a
# 24-partial sweep over 1-partial duals (48 scalars per value, versus 625 for a
# full 24x24 `ForwardDiff.hessian`), both scattered into the 3549-long result.
# Differentiating the whole sum at once would mean 3549 partials.
#
# WHAT IS PRECOMPUTED. The shape-function gradients `G(vertex, dir)` and the
# taper constants `0.45/V^2`, `3.1*V^2`, `3*log(V^2)` depend only on the
# element's parameters `(RI,RJ,RK)` and on the globals -- not on the
# variables. They are built once per element per quadrature point (24 + 3
# numbers, in the element's working precision) and held in `JimackElement`.
# What remains inside the AD unit is exactly the part that sees the variables.
#
# WHY `jimack_pointwise` IS `@noinline`. It is the AD unit's inner routine,
# called twice per element from three different closures (value, gradient,
# hvp) and instantiated for plain `T`, for `Dual{_,T,24}` and for
# `Dual{_,Dual{_,T,1},24}`. One compiled copy per dual type instead of two is
# a code-size decision only; the function stays allocation-free and
# branch-free on values, hence callable from a GPU kernel.
#
# ON COMPARING THE HESSIAN. JIMACK's classification is `OUR2-AN-3549-0` -- 4th
# character `2`, and the SIF's `INDIVIDUALS` block does supply a full `H`
# section -- so CUTEst's Hessian is a real one, unlike SARO's, and is compared
# directly. As `hprod`, since the matrix is 3549x3549: four random directions at
# each of two points agree to 4.1e-16 (at x0) and 2.5e-14 .. 1.2e-13 (at a
# perturbed point, where |Hv| ~ 1e7 and both sides sum 864 cancelling element
# blocks). This port's sparsity pattern independently has exactly CUTEst's
# `nnzh = 118824`. Two further checks that do not go through CUTEst's Hessian at
# all: a central difference of CUTEst's own gradient, which both sides miss by
# the SAME amount to 4 significant digits; and the rigid TRANSLATION direction,
# a null vector of the exact Hessian because the energy sees only grad u
# (|H*1| = 2.0e-15 here, 2.4e-13 for CUTEst, against |H*v| ~ 2e3).

# The 8 vertices of a cell, in the order the Fortran sums them:
#
#     DU(UX,I) = AXLLL*G(1,1,1,I) + AXLLU*G(1,1,2,I) + AXLUL*G(1,2,1,I) + ...
#
# i.e. name-lexicographic with L < U, which is z fastest and x slowest. This is
# NOT the SIF's `EV INTEGRAL` declaration order (that one is x fastest), so the
# 24 variables are gathered by NAME rather than by position.
const JIMACK_VERTICES = (:LLL, :LLU, :LUL, :LUU, :ULL, :ULU, :UUL, :UUU)

"""
    JIMACK_EVARS

The 24 elemental variables of SIF element type `INTEGRAL`, ordered
`(AX, AY, AZ) x (LLL, LLU, LUL, LUU, ULL, ULU, UUL, UUU)` -- the order in
which the Fortran accumulates `grad u`, so that a Float64 evaluation
reproduces its rounding and not merely its value.
"""
const JIMACK_EVARS = (
    :AXLLL, :AXLLU, :AXLUL, :AXLUU, :AXULL, :AXULU, :AXUUL, :AXUUU,
    :AYLLL, :AYLLU, :AYLUL, :AYLUU, :AYULL, :AYULU, :AYUUL, :AYUUU,
    :AZLLL, :AZLLU, :AZLUL, :AZLUU, :AZULL, :AZULU, :AZUUL, :AZUUU,
)

"""
    JimackElement{T}

Everything one SIF `INTEGRAL` element needs that does NOT depend on the
variables.

`idx` holds the global index of each of [`JIMACK_EVARS`](@ref). `g1`/`g2` are
the shape-function gradients `G(vertex, dir)` at the two distinct quadrature
points, laid out as `(GX[1..8], GY[1..8], GZ[1..8])` over the vertices of
[`JIMACK_VERTICES`](@ref). `c1`/`c2` are the quadrature point's taper
constants `(0.45/V^2, 3.1*V^2, 3*log(V^2))`.

Two quadrature points, not eight: see the header for the `K1` bug that makes
the Fortran visit each of them four times.
"""
struct JimackElement{T}
    idx::NTuple{24,Int}
    g1::NTuple{24,T}
    g2::NTuple{24,T}
    c1::NTuple{3,T}
    c2::NTuple{3,T}
end

# ---------------------------------------------------------------------------
# The value, as a plain generic function

# One `DU(axis, dir)`: the 8-term sum the Fortran writes out longhand, with
# `oa` the axis block offset into the nodal values and `og` the direction
# block offset into G. Both are literals at every call site.
@inline _jimack_du(a::NTuple{24,T}, oa::Int, g::NTuple{24,S}, og::Int) where {S,T} =
    (a[oa + 1] * g[og + 1] + a[oa + 2] * g[og + 2] +
     a[oa + 3] * g[og + 3] + a[oa + 4] * g[og + 4] +
     a[oa + 5] * g[og + 5] + a[oa + 6] * g[og + 6] +
     a[oa + 7] * g[og + 7] + a[oa + 8] * g[og + 8])

"""
    jimack_pointwise(g, c, a) -> scalar

`IPHI`: the pointwise stored energy at one quadrature point, from the 24 nodal
values `a` and the precomputed `g` (shape-function gradients) and `c` (taper
constants) of a [`JimackElement`](@ref).

Generic in both types: `g`/`c` may be in the working precision while `a` is a
`ForwardDiff.Dual` over it. Allocation-free and branch-free on values. Literals
and operation order follow the Fortran, so a Float64 evaluation reproduces its
bit pattern.

`@noinline` on purpose -- see the header.
"""
@noinline function jimack_pointwise(g::NTuple{24,S}, c::NTuple{3,S},
                                   a::NTuple{24,T}) where {S,T}
    # grad u: rows AX/AY/AZ (the deformed coordinate), columns the three
    # spatial directions of G.
    uxx = _jimack_du(a, 0, g, 0)
    uxy = _jimack_du(a, 0, g, 8)
    uxz = _jimack_du(a, 0, g, 16)
    uyx = _jimack_du(a, 8, g, 0)
    uyy = _jimack_du(a, 8, g, 8)
    uyz = _jimack_du(a, 8, g, 16)
    uzx = _jimack_du(a, 16, g, 0)
    uzy = _jimack_du(a, 16, g, 8)
    uzz = _jimack_du(a, 16, g, 16)

    # F = (grad u)(grad u)^T
    fxx = uxx * uxx + uxy * uxy + uxz * uxz
    fxy = uxx * uyx + uxy * uyy + uxz * uyz
    fyy = uyx * uyx + uyy * uyy + uyz * uyz
    fxz = uxx * uzx + uxy * uzy + uxz * uzz
    fyz = uyx * uzx + uyy * uzy + uyz * uzz
    fzz = uzx * uzx + uzy * uzy + uzz * uzz

    # the three principal invariants, in the Fortran's groupings
    fxxpyy = fxx + fyy
    fxxyy = fxx * fyy - fxy * fxy
    fxxzz = fxx * fzz - fxz * fxz
    fyyzz = fyy * fzz - fyz * fyz
    fxyyz = fxy * fyz - fyy * fxz
    fxyxz = fxy * fxz - fxx * fyz
    inv1 = fxxpyy + fzz
    inv2 = fxxyy + fyyzz + fxxzz
    inv3 = fzz * fxxyy + fxz * fxyyz + fyz * fxyxz

    bv2, cv2, l3v2 = c
    return S(0.65) * inv1 + bv2 * inv2 - cv2 * (log(inv3) - l3v2)
end

"""
    jimack_element_value(e, a) -> scalar

`IEINT`: the value of SIF element `e` at its 24 nodal values `a`.

The sum of the Fortran's eight `P(L1)`, which is four copies of each of the
two distinct quadrature points, accumulated from zero in that order.
"""
@inline function jimack_element_value(e::JimackElement{S}, a::NTuple{24,T}) where {S,T}
    p1 = jimack_pointwise(e.g1, e.c1, a)
    p2 = jimack_pointwise(e.g2, e.c2, a)
    s = zero(T)
    s += p1
    s += p1
    s += p1
    s += p1
    s += p2
    s += p2
    s += p2
    s += p2
    return s
end

# ---------------------------------------------------------------------------
# Derivatives, by AD over that one function

@inline _jimack_seed(::Type{V}, i::Int) where {V} =
    ForwardDiff.Partials(ntuple(j -> ifelse(i == j, one(V), zero(V)), Val(24)))

"""
    jimack_element_grad(e, a) -> Partials{24}

The 24 partial derivatives of [`jimack_element_value`](@ref), one forward
sweep.
"""
@inline function jimack_element_grad(e::JimackElement{S}, a::NTuple{24,T}) where {S,T}
    D = ForwardDiff.Dual{:JIMACKg,T,24}
    ad = ntuple(i -> D(a[i], _jimack_seed(T, i)), Val(24))
    return ForwardDiff.partials(jimack_element_value(e, ad))
end

"""
    jimack_element_hvp(e, a, v) -> NTuple{24}

The element Hessian times `v`, as `d/de grad f(a + e*v)`: a 24-partial forward
sweep over 1-partial duals, i.e. 48 scalars per value instead of the 625 a full
`ForwardDiff.hessian` of a 24-variable function would carry.
"""
@inline function jimack_element_hvp(e::JimackElement{S}, a::NTuple{24,T},
                                    v::NTuple{24,T}) where {S,T}
    Di = ForwardDiff.Dual{:JIMACKv,T,1}
    Do = ForwardDiff.Dual{:JIMACKh,Di,24}
    ad = ntuple(i -> Do(Di(a[i], ForwardDiff.Partials((v[i],))), _jimack_seed(Di, i)),
                Val(24))
    p = ForwardDiff.partials(jimack_element_value(e, ad))
    return ntuple(i -> ForwardDiff.partials(p[i], 1), Val(24))
end

# ---------------------------------------------------------------------------
# Building the per-element geometry

"""
    jimack_gauss(sgn, ...) -> (g, c)

The shape-function gradients and taper constants of one quadrature point: the
`PSI` calls of `IPHI` plus the `V`-dependent scalars of its energy, for the
cell-centre offset `sgn * (DELX, DELY, DELZ)` with `sgn = -1` or `+1`.

`g` is ordered `(GX[1..8], GY[1..8], GZ[1..8])` over
[`JIMACK_VERTICES`](@ref). `PSI` ignores the vertex coordinates it is handed
and uses only the SIGNED cell widths (`DX` at a lower vertex, `-DX` at an
upper one) and the scaled origins `SXL = XL/DX` / `SXU = -XU/DX`, which is why
only those appear here.
"""
function jimack_gauss(sgn::Int, xmid::T, ymid::T, zmid::T,
                      delx::T, dely::T, delz::T, dx::T, dy::T, dz::T,
                      sxl::T, sxu::T, syl::T, syu::T, szl::T, szu::T,
                      epsv::T, delta::T) where {T}
    x = sgn < 0 ? xmid - delx : xmid + delx
    y = sgn < 0 ? ymid - dely : ymid + dely
    z = sgn < 0 ? zmid - delz : zmid + delz

    one_ = one(T)
    gx = Vector{T}(undef, 8)
    gy = Vector{T}(undef, 8)
    gz = Vector{T}(undef, 8)
    for v in 1:8
        # name order: x slowest, z fastest; 0 = lower (L), 1 = upper (U)
        vi = ((v - 1) >> 2) & 1
        vj = ((v - 1) >> 1) & 1
        vk = (v - 1) & 1
        dxv = vi == 0 ? dx : -dx
        dyv = vj == 0 ? dy : -dy
        dzv = vk == 0 ? dz : -dz
        sxv = vi == 0 ? sxl : sxu
        syv = vj == 0 ? syl : syu
        szv = vk == 0 ? szl : szu

        ex = x / dxv
        ey = y / dyv
        ez = z / dzv
        axy = one_ + szv
        axz = one_ + syv
        ayz = one_ + sxv
        ax = axy * axz
        ay = axy * ayz
        az = axz * ayz
        gx[v] = (-ax + axy * ey + axz * ez - ey * ez) / dxv
        gy[v] = (-ay + axy * ex + ayz * ez - ex * ez) / dyv
        gz[v] = (-az + axz * ex + ayz * ey - ex * ey) / dzv
    end

    vv = one_ - epsv * z / delta
    v2 = vv * vv
    c = (T(0.45) / v2, T(3.1) * v2, T(3) * log(v2))
    g = ntuple(t -> t <= 8 ? gx[t] : t <= 16 ? gy[t - 8] : gz[t - 16], Val(24))
    return g, c
end

"""
    jimack_elements(P, T; n, m, delta, epsv, w) -> Vector{JimackElement{T}}

Build one [`JimackElement`](@ref) per deferred `INTEGRAL` element of a
`sif_decode(...; external = :defer)` problem, in SIF group order.

`n`, `m` are the grid line counts and `delta`, `epsv`, `w` the three GLOBALS
constants in whichever reading the caller chose (see `jimack_model`). The
cell geometry follows `IEINT` statement for statement, including the
element-independent `DZ = DELTA/RM` and `DELZ = HALFW*DELTA/RM` that the SIF
recomputes inside `INDIVIDUALS`.
"""
function jimack_elements(P, ::Type{T}, elemnames; n::Int, m::Int,
                         delta::T, epsv::T, w::T) where {T}
    one_ = one(T)
    half = T(0.5)
    rm = T(m)
    rn = T(n)
    dx = T(2) / rn
    dy = dx
    halfw = half * w
    delx = w / rn
    dely = dx                      # the Fortran's `A DELY DX`; see the header
    dz = delta / rm
    delz = halfw * delta / rm

    els = Vector{JimackElement{T}}(undef, length(elemnames))
    for (p, ename) in enumerate(elemnames)
        e = P.elements[ename]
        # `A I INT( RI )` on `RI = I + 0.1`
        i = trunc(Int, e.parvals[:RI])
        j = trunc(Int, e.parvals[:RJ])
        k = trunc(Int, e.parvals[:RK])
        (0 <= i < n && 0 <= j < n && 0 <= k < m) ||
            error("JIMACK: element $ename has (I,J,K) = ($i,$j,$k), outside " *
                  "the $(n)x$(n)x$(m) grid")
        idx = ntuple(t -> e.varmap[JIMACK_EVARS[t]], Val(24))
        length(unique(idx)) == 24 ||
            error("JIMACK: element $ename does not map its 24 elemental " *
                  "variables to 24 distinct problem variables")

        xl = -one_ + dx * T(i)
        yl = -one_ + dy * T(j)
        zl = dz * T(k)
        xu = xl + dx
        yu = yl + dy
        zu = zl + dz
        xmid = half * (xl + xu)
        ymid = half * (yl + yu)
        zmid = half * (zl + zu)
        sxl = xl / dx
        syl = yl / dy
        szl = zl / dz
        sxu = -xu / dx
        syu = -yu / dy
        szu = -zu / dz

        g1, c1 = jimack_gauss(-1, xmid, ymid, zmid, delx, dely, delz, dx, dy, dz,
                              sxl, sxu, syl, syu, szl, szu, epsv, delta)
        g2, c2 = jimack_gauss(+1, xmid, ymid, zmid, delx, dely, delz, dx, dy, dz,
                              sxl, sxu, syl, syu, szl, szu, epsv, delta)
        els[p] = JimackElement{T}(idx, g1, g2, c1, c2)
    end
    return els
end

# ---------------------------------------------------------------------------
# Assembly

"""
    jimack_model(; T = Float64, backend = nothing,
                   fortran_literals = :single) -> ExaModel

Build JIMACK as an `ExaModel` whose objective callbacks are the generic Julia
port of `IEINT`/`IPHI` rather than the compiled Fortran.

Unconstrained, 3549 variables, no bounds. The objective is the sum of the 864
element energies divided by the SIF group scale `2*M*N^2/DELTA = 17280`, and
goes in as a single `ScalarNonlinearOracle` with explicit Hessian sparsity --
the union of the 864 lower-triangular 24x24 element blocks, deduplicated,
rather than a dense 3549x3549 triangle. Derivatives are ForwardDiff over
[`jimack_element_value`](@ref), one element at a time; the Fortran's own
gradient and Hessian arrays are not ported.

Needs no `sifdecoder` and no Fortran toolchain: the structure and the grid
come from `sif_decode(...; external = :defer)`.

`fortran_literals` selects how the three bare GLOBALS constants `DELTA = 0.1`,
`EPS = 0.01` and `W = 0.577350269189` are read. `:single` (the default) is
what gfortran compiles -- a bare Fortran real literal is default REAL, so each
is rounded through Float32 before widening -- and reproduces CUTEst's objective
bit for bit. `:double` is the naive reading and is wrong by ~1.5e-8 relative;
it is kept only so the two can be compared. See the header.

Unlike the older native oracle models, both NLPModels Hessian entry points
work here: measured, `hprod(m, x, v)` and `hprod(m, x, Float64[], v)` return
the same vector to the last bit on this ExaModels (0.12.1 delegates the no-`y`
forms to the `y`-taking ones with a zero multiplier, which the `BA-L*` and
`AKIVA`/`HIELOW` docstrings predate). `T = Float32` also builds and evaluates
`obj`, `grad` and `hprod`; note that ExaModels' scalar-oracle work buffers are
`Vector{Float64}` regardless, so the Hessian path is computed in Float64 even
then. [`jimack_pointwise`](@ref) itself is fully generic -- Float32 and
BigFloat evaluations are part of the verification.

!!! note "Materializing the Hessian is expensive, `hprod` is not"
    `ScalarNonlinearOracle` takes `hvp!`, not an explicit `hess!`, so
    `hess_coord!` reconstructs the objective block one COLUMN at a time -- 3549
    `hvp!` calls here. Explicit sparsity shrinks the stored pattern (and the
    consumer's work) but not that column count. `hprod` is one `hvp!` call and
    is the right way to probe this model.

    Measured per call on shin-compute-001: `obj` 0.13 ms, `grad!` 1.2 ms,
    `hprod!` 5.1 ms, against CUTEst's 13 ms and 27 ms for `obj` and `grad` --
    the port is ~20x faster than the Fortran on both, because the Fortran
    recomputes the shape-function gradients and the whole hand-coded Hessian on
    every element call. `grad!` and `hprod!` allocate 128 and 224 bytes (the
    `ntuple` closures' boxes), independent of the problem size.
"""
function jimack_model(; T = Float64, backend = nothing,
                      fortran_literals::Symbol = :single)
    # The oracle's callbacks run on the CPU, so a backend is refused rather
    # than ignored: accepting one would misreport where the model lives.
    # `jimack_pointwise` is type-generic, allocation-free and branch-free on
    # values, so a device path needs the oracle wiring and not a rewrite.
    backend === nothing || error(
        "jimack_model does not take a backend: the port's callbacks run on " *
        "the CPU. jimack_pointwise itself is GPU-safe.")

    P = ExaModelsSIF.sif_decode(sif_problem_path("JIMACK"); external = :defer)
    P.external_types == Dict("INTEGRAL" => :IEINT) || error(
        "JIMACK: expected a single deferred INTEGRAL element type calling " *
        "IEINT, found $(P.external_types)")
    all(==(1.0), P.varscale) || error("JIMACK: variable scaling is not supported here")
    length(P.grouporder) == 1 ||
        error("JIMACK: expected 1 group, found $(length(P.grouporder))")
    gn = only(P.grouporder)
    g = P.groups[gn]
    (g.kind == 'N' && isempty(g.lin) && g.constant == 0.0 && g.range_ === nothing &&
     isempty(g.gparvals)) || error("JIMACK: unexpected objective group shape")
    all(==(1.0), last.(g.elems)) ||
        error("JIMACK: objective group weights its elements by " *
              "$(unique(last.(g.elems))); expected all 1")
    elemnames = first.(g.elems)

    # The grid, from the elements rather than from the Fortran's PARAMETER
    # statement, so a re-parameterized instance is caught instead of silently
    # mis-meshed. `(I,J,K)` runs over 0:N-1 x 0:N-1 x 0:M-1.
    ijk = [(trunc(Int, P.elements[en].parvals[:RI]),
            trunc(Int, P.elements[en].parvals[:RJ]),
            trunc(Int, P.elements[en].parvals[:RK])) for en in elemnames]
    n = maximum(first, ijk) + 1
    nj = maximum(t -> t[2], ijk) + 1
    m = maximum(last, ijk) + 1
    n == nj || error("JIMACK: the grid is $n x $nj in x,y; the Fortran assumes N x N")
    nvar = length(P.varnames)
    (length(elemnames) == n * n * m && nvar == 3 * (n + 1)^2 * (m + 1)) || error(
        "JIMACK: $(length(elemnames)) elements and $nvar variables do not match " *
        "an $(n)x$(n)x$(m) grid")

    # DELTA appears twice: as a proper double in the DATA sections (where it
    # sets this scale) and as a bare literal in the Fortran GLOBALS. The SIF
    # says in both places that they must agree; check it, then round the
    # Fortran's copy.
    delta_data = 2 * m * n^2 / g.scale
    isapprox(delta_data, 0.1; rtol = 1e-12) || error(
        "JIMACK: the group scale implies DELTA = $delta_data, but the appended " *
        "Fortran hard-codes 0.1; the SIF requires the two to agree")

    lit = if fortran_literals === :single
        # what gfortran does with a bare real literal
        v -> T(Float64(Float32(v)))
    elseif fortran_literals === :double
        v -> T(v)
    else
        error("JIMACK: fortran_literals must be :single or :double, got " *
              "$(repr(fortran_literals))")
    end
    delta = lit(0.1)
    epsv = lit(0.01)
    w = lit(0.577350269189)

    els = jimack_elements(P, T, elemnames; n, m, delta, epsv, w)
    scale = T(g.scale)

    # Hessian sparsity: the union of the elements' lower-triangular 24x24
    # blocks. Deduplicated, because hess_coord! ASSIGNS each slot from the hvp
    # result rather than accumulating, so a repeated (row, col) would be
    # double counted downstream.
    pairs = Set{Tuple{Int,Int}}()
    for e in els, t1 in 1:24, t2 in 1:24
        i, j = e.idx[t1], e.idx[t2]
        i >= j && push!(pairs, (i, j))
    end
    sp = sort!(collect(pairs))
    hr = first.(sp)
    hc = last.(sp)

    f = function (x)
        E = eltype(x)
        s = zero(E)
        for e in els
            s += jimack_element_value(e, ntuple(t -> x[e.idx[t]], Val(24)))
        end
        return s / E(scale)
    end

    grad! = function (gv, x)
        E = eltype(gv)
        fill!(gv, zero(E))
        for e in els
            gp = jimack_element_grad(e, ntuple(t -> E(x[e.idx[t]]), Val(24)))
            for t in 1:24
                gv[e.idx[t]] += gp[t]
            end
        end
        gv ./= E(scale)
        return gv
    end

    hvp! = function (Hv, x, v)
        E = eltype(Hv)
        fill!(Hv, zero(E))
        for e in els
            h = jimack_element_hvp(e, ntuple(t -> E(x[e.idx[t]]), Val(24)),
                                   ntuple(t -> E(v[e.idx[t]]), Val(24)))
            for t in 1:24
                Hv[e.idx[t]] += h[t]
            end
        end
        Hv ./= E(scale)
        return Hv
    end

    c = ExaModels.ExaCore(T)
    c, _ = ExaModels.add_var(c, nvar; start = P.x0, lvar = P.lvar, uvar = P.uvar)
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = nvar, f = f, grad! = grad!, hvp! = hvp!,
        hess_rows = hr, hess_cols = hc, adapt = Val(true)))
    return ExaModels.ExaModel(c)
end
