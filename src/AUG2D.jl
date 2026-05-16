# AUG2D / AUG2DC / AUG2DQP / AUG2DCQP
#
# Expanded system formulation of a 2-D PDE system (Laplace eq on NX×NY grid).
# Pure linear EQP: minimize 0.5*||x_obj||^2 s.t. divergence equations.
# SQUARE group type: F(alpha) = 0.5*alpha^2.
# Constraint RHS = 1.0 (DEFAULT constant in SIF).
# SIF: Nick Gould, February 1994.  classification QLR2-AN-V-V
#
# Variable ordering matches SIF VARIABLES declaration order:
#   Inner (j=1:NY-1, i=1:NX-1): X(i,j) then Y(i,j) alternating
#   X(i,NY) for i=1:NX-1
#   Y(NX,j) for j=1:NY-1
#   [AUG2DC: X(i,0), X(i,NY+1) for i=1:NX; Y(0,j), Y(NX+1,j) for j=1:NY (also objective)]
#   [AUG2D:  same border vars but without objective groups, appended after]
#
# Constraints V(I,J) for J=1:NY (outer), I=1:NX (inner):
#   X(I,J) - X(I-1,J) + Y(I,J) - Y(I,J-1) + border_terms - 1 = 0

function _aug2d_make_model(NX, NY; border_in_obj, lb_top, lb_right, T, backend, kwargs...)
    NXm1 = NX - 1
    NYm1 = NY - 1

    # ── Variable index mapping ────────────────────────────────────────────────
    n_inner   = NXm1 * NYm1
    off_xNY   = 2 * n_inner
    off_yNX   = off_xNY + NXm1
    off_xb0   = off_yNX + NYm1
    off_xbNYp = off_xb0   + NX
    off_yb0   = off_xbNYp + NX
    off_ybNXp = off_yb0   + NY
    nvar      = off_ybNXp + NY

    X_inn(i, j) = 2 * ((j - 1) * NXm1 + (i - 1)) + 1   # i=1:NXm1, j=1:NYm1
    Y_inn(i, j) = 2 * ((j - 1) * NXm1 + (i - 1)) + 2   # i=1:NXm1, j=1:NYm1
    X_top(i)    = off_xNY   + i                          # i=1:NXm1
    Y_rgt(j)    = off_yNX   + j                          # j=1:NYm1
    xb0(i)      = off_xb0   + i                          # i=1:NX
    xbNYp(i)    = off_xbNYp + i                          # i=1:NX
    yb0(j)      = off_yb0   + j                          # j=1:NY
    ybNXp(j)    = off_ybNXp + j                          # j=1:NY

    X(i, j) = j < NY ? X_inn(i, j) : X_top(i)   # i=1:NXm1, j=1:NY
    Y(i, j) = i < NX ? Y_inn(i, j) : Y_rgt(j)   # i=1:NX,   j=1:NYm1

    # ── Bounds ────────────────────────────────────────────────────────────────
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf),  nvar)
    if lb_top !== nothing
        for i in 1:NXm1; lvar[X_top(i)] = T(lb_top); end
    end
    if lb_right !== nothing
        for j in 1:NYm1; lvar[Y_rgt(j)] = T(lb_right); end
    end
    start = zeros(T, nvar)

    # ── ExaCore ───────────────────────────────────────────────────────────────
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # ── Objective (SQUARE: 0.5 * v^2) ────────────────────────────────────────
    xinn_data = [(idx=Int32(X_inn(i,j)),) for j in 1:NYm1 for i in 1:NXm1]
    yinn_data = [(idx=Int32(Y_inn(i,j)),) for j in 1:NYm1 for i in 1:NXm1]
    ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in xinn_data)
    ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in yinn_data)
    xtop_data = [(idx=Int32(X_top(i)),) for i in 1:NXm1]
    yrgt_data = [(idx=Int32(Y_rgt(j)),) for j in 1:NYm1]
    ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in xtop_data)
    ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in yrgt_data)
    if border_in_obj
        xb0_data   = [(idx=Int32(xb0(i)),)   for i in 1:NX]
        xbNYp_data = [(idx=Int32(xbNYp(i)),) for i in 1:NX]
        yb0_data   = [(idx=Int32(yb0(j)),)   for j in 1:NY]
        ybNXp_data = [(idx=Int32(ybNXp(j)),) for j in 1:NY]
        ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in xb0_data)
        ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in xbNYp_data)
        ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in yb0_data)
        ExaModels.@add_obj(c, T(0.5) * x[d.idx]^2 for d in ybNXp_data)
    end

    # ── Constraints (J outer, I inner: matches CUTEst ordering) ──────────────
    con_data = [
        let
            xp   = (I <= NXm1) ? Int32(X(I, J))   : Int32(1)
            cxp  = (I <= NXm1) ? T(1) : T(0)
            xm   = (I >= 2)    ? Int32(X(I-1, J)) : Int32(1)
            cxm  = (I >= 2)    ? T(1) : T(0)
            yp   = (J <= NYm1) ? Int32(Y(I, J))   : Int32(1)
            cyp  = (J <= NYm1) ? T(1) : T(0)
            ym   = (J >= 2)    ? Int32(Y(I, J-1)) : Int32(1)
            cym  = (J >= 2)    ? T(1) : T(0)
            eb0  = (J == 1)    ? Int32(xb0(I))    : Int32(1)
            ceb0 = (J == 1)    ? T(1) : T(0)
            ebT  = (J == NY)   ? Int32(xbNYp(I))  : Int32(1)
            cebT = (J == NY)   ? T(1) : T(0)
            el   = (I == 1)    ? Int32(yb0(J))    : Int32(1)
            cel  = (I == 1)    ? T(1) : T(0)
            er   = (I == NX)   ? Int32(ybNXp(J))  : Int32(1)
            cer  = (I == NX)   ? T(1) : T(0)
            (xp=xp, cxp=cxp, xm=xm, cxm=cxm,
             yp=yp, cyp=cyp, ym=ym, cym=cym,
             eb0=eb0, ceb0=ceb0, ebT=ebT, cebT=cebT,
             el=el, cel=cel, er=er, cer=cer)
        end
        for J in 1:NY for I in 1:NX
    ]

    ExaModels.@add_con(c,
        d.cxp  * x[d.xp]  - d.cxm  * x[d.xm]  +
        d.cyp  * x[d.yp]  - d.cym  * x[d.ym]  +
        d.ceb0 * x[d.eb0] + d.cebT * x[d.ebT] +
        d.cel  * x[d.el]  + d.cer  * x[d.er]  - T(1)
        for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end

function aug2d_model(NX = 100, NY = 100; T = Float64, backend = nothing, kwargs...)
    _aug2d_make_model(NX, NY; border_in_obj = false,
        lb_top = nothing, lb_right = nothing, T = T, backend = backend, kwargs...)
end

function aug2dc_model(NX = 100, NY = 100; T = Float64, backend = nothing, kwargs...)
    _aug2d_make_model(NX, NY; border_in_obj = true,
        lb_top = nothing, lb_right = nothing, T = T, backend = backend, kwargs...)
end

function aug2dqp_model(NX = 100, NY = 100; T = Float64, backend = nothing, kwargs...)
    _aug2d_make_model(NX, NY; border_in_obj = false,
        lb_top = 1.0, lb_right = 1.0, T = T, backend = backend, kwargs...)
end

function aug2dcqp_model(NX = 100, NY = 100; T = Float64, backend = nothing, kwargs...)
    _aug2d_make_model(NX, NY; border_in_obj = true,
        lb_top = 1.0, lb_right = 1.0, T = T, backend = backend, kwargs...)
end
