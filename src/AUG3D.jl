# AUG3D / AUG3DC / AUG3DQP / AUG3DCQP
#
# Expanded system formulation of a 3-D PDE system (Laplace eq on NX×NY×NZ grid).
# Pure linear EQP: minimize 0.5*||x_obj||^2 s.t. divergence equations.
# SQUARE group type: F(alpha) = 0.5*alpha^2.
# Constraint RHS = 1.0 (DEFAULT constant in SIF).
# SIF: Nick Gould, February 1994.  classification QLR2-AN-V-V
#
# Variable ordering follows SIF VARIABLES declaration:
#   Inner (K=1:NZ-1, J=1:NY-1, I=1:NX-1): X(I,J,K), Y(I,J,K), Z(I,J,K) interleaved
#   Face-x (K=1:NZ-1, J=1:NY-1): Y(NX,J,K), Z(NX,J,K)
#   Face-y (K=1:NZ-1, I=1:NX-1): X(I,NY,K), Z(I,NY,K)
#   Face-z (J=1:NY-1, I=1:NX-1): X(I,J,NZ), Y(I,J,NZ)
#   [then AUG3DC: edge/border vars with objective groups]
#   [AUG3D: same border constraint-only vars appended after]
#
# Border constraint variables (edge flows):
#   Y(0,J,K), Z(0,J,K), Y(NX+1,J,K), Z(NX+1,J,K) for K=1:NZ, J=1:NY
#   X(I,0,K), Z(I,0,K), X(I,NY+1,K), Z(I,NY+1,K) for K=1:NZ, I=1:NX
#   X(I,J,0), Y(I,J,0), X(I,J,NZ+1), Y(I,J,NZ+1) for J=1:NY, I=1:NX
#
# Constraints V(I,J,K) for K outer, J middle, I inner: I=1:NX, J=1:NY, K=1:NZ
#   X(I,J,K) - X(I-1,J,K) + Y(I,J,K) - Y(I,J-1,K) + Z(I,J,K) - Z(I,J,K-1) + borders = 1

function _aug3d_make_model(NX, NY, NZ; border_in_obj, lb_xface, lb_yface, lb_zface,
                            T, backend, kwargs...)
    NXm1 = NX - 1; NYm1 = NY - 1; NZm1 = NZ - 1

    # ── Variable layout ───────────────────────────────────────────────────────
    # Inner block: K=1:NZm1, J=1:NYm1, I=1:NXm1 → 3 vars per cell: X,Y,Z
    n_inner  = NXm1 * NYm1 * NZm1
    off_face_yz = 3 * n_inner                # Y(NX,J,K), Z(NX,J,K): K=1:NZm1, J=1:NYm1
    n_facex   = NYm1 * NZm1                  # pairs Y,Z on x=NX face
    off_face_xz = off_face_yz + 2 * n_facex  # X(I,NY,K), Z(I,NY,K): K=1:NZm1, I=1:NXm1
    n_facey   = NXm1 * NZm1
    off_face_xy = off_face_xz + 2 * n_facey  # X(I,J,NZ), Y(I,J,NZ): J=1:NYm1, I=1:NXm1
    n_facez   = NXm1 * NYm1
    off_bord  = off_face_xy + 2 * n_facez    # start of border/edge vars

    # Border variable groups (4 per face pair, 3 face pairs):
    # K×J: Y(0,J,K), Z(0,J,K), Y(NX+1,J,K), Z(NX+1,J,K) for K=1:NZ, J=1:NY
    n_KJ = NZ * NY
    off_y0jk   = off_bord
    off_z0jk   = off_y0jk  + n_KJ
    off_yNXpjk = off_z0jk  + n_KJ
    off_zNXpjk = off_yNXpjk + n_KJ
    # K×I: X(I,0,K), Z(I,0,K), X(I,NX+1,K), Z(I,NYp,K) for K=1:NZ, I=1:NX
    n_KI = NZ * NX
    off_xi0k   = off_zNXpjk + n_KJ
    off_zi0k   = off_xi0k   + n_KI
    off_xiNYpk = off_zi0k   + n_KI
    off_ziNYpk = off_xiNYpk + n_KI
    # J×I: X(I,J,0), Y(I,J,0), X(I,J,NZ+1), Y(I,J,NZ+1) for J=1:NY, I=1:NX
    n_JI = NY * NX
    off_xij0   = off_ziNYpk + n_KI
    off_yij0   = off_xij0   + n_JI
    off_xijNZp = off_yij0   + n_JI
    off_yijNZp = off_xijNZp + n_JI
    nvar       = off_yijNZp + n_JI

    # Index helpers for inner/face vars
    # inner: K=1:NZm1, J=1:NYm1, I=1:NXm1
    cell(I,J,K) = (K-1)*NYm1*NXm1 + (J-1)*NXm1 + (I-1)   # 0-based cell index
    Xinn(I,J,K)  = 3*cell(I,J,K) + 1
    Yinn(I,J,K)  = 3*cell(I,J,K) + 2
    Zinn(I,J,K)  = 3*cell(I,J,K) + 3
    # face_yz (I=NX): K=1:NZm1, J=1:NYm1
    fyz(J,K)  = off_face_yz + 2*((K-1)*NYm1 + (J-1))
    YNX(J,K)   = fyz(J,K) + 1
    ZNX(J,K)   = fyz(J,K) + 2
    # face_xz (J=NY): K=1:NZm1, I=1:NXm1
    fxz(I,K)  = off_face_xz + 2*((K-1)*NXm1 + (I-1))
    XNY(I,K)   = fxz(I,K) + 1
    ZNY(I,K)   = fxz(I,K) + 2
    # face_xy (K=NZ): J=1:NYm1, I=1:NXm1
    fxy(I,J)  = off_face_xy + 2*((J-1)*NXm1 + (I-1))
    XNZ(I,J)   = fxy(I,J) + 1
    YNZ(I,J)   = fxy(I,J) + 2
    # border vars
    Y0jk(J,K)    = off_y0jk   + (K-1)*NY + (J-1) + 1
    Z0jk(J,K)    = off_z0jk   + (K-1)*NY + (J-1) + 1
    YNXpjk(J,K)  = off_yNXpjk + (K-1)*NY + (J-1) + 1
    ZNXpjk(J,K)  = off_zNXpjk + (K-1)*NY + (J-1) + 1
    Xi0k(I,K)    = off_xi0k   + (K-1)*NX + (I-1) + 1
    Zi0k(I,K)    = off_zi0k   + (K-1)*NX + (I-1) + 1
    XiNYpk(I,K)  = off_xiNYpk + (K-1)*NX + (I-1) + 1
    ZiNYpk(I,K)  = off_ziNYpk + (K-1)*NX + (I-1) + 1
    Xij0(I,J)    = off_xij0   + (J-1)*NX + (I-1) + 1
    Yij0(I,J)    = off_yij0   + (J-1)*NX + (I-1) + 1
    XijNZp(I,J)  = off_xijNZp + (J-1)*NX + (I-1) + 1
    YijNZp(I,J)  = off_yijNZp + (J-1)*NX + (I-1) + 1

    # Unified access X(I,J,K): I=1:NXm1, any J∈1:NY, any K∈1:NZ
    function X(I,J,K)
        if K < NZ && J < NY
            Xinn(I,J,K)
        elseif K == NZ && J < NY
            XNZ(I,J)
        elseif K < NZ && J == NY
            XNY(I,K)
        else
            error("X($I,$J,$K) undefined")
        end
    end
    function Y(I,J,K)
        if I < NX && K < NZ && J < NY
            Yinn(I,J,K)
        elseif I == NX && K < NZ && J < NY
            YNX(J,K)
        elseif I < NX && K == NZ && J < NY
            YNZ(I,J)
        else
            error("Y($I,$J,$K) undefined")
        end
    end
    function Z(I,J,K)
        if I < NX && J < NY && K < NZ
            Zinn(I,J,K)
        elseif I == NX && J < NY && K < NZ
            ZNX(J,K)
        elseif I < NX && J == NY && K < NZ
            ZNY(I,K)
        else
            error("Z($I,$J,$K) undefined")
        end
    end

    # ── Bounds ────────────────────────────────────────────────────────────────
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf),  nvar)
    if lb_xface !== nothing
        for K in 1:NZm1, J in 1:NYm1
            lvar[YNX(J,K)] = T(lb_xface)
            lvar[ZNX(J,K)] = T(lb_xface)
        end
    end
    if lb_yface !== nothing
        for K in 1:NZm1, I in 1:NXm1
            lvar[XNY(I,K)] = T(lb_yface)
            lvar[ZNY(I,K)] = T(lb_yface)
        end
    end
    if lb_zface !== nothing
        for J in 1:NYm1, I in 1:NXm1
            lvar[XNZ(I,J)] = T(lb_zface)
            lvar[YNZ(I,J)] = T(lb_zface)
        end
    end
    start = zeros(T, nvar)

    # ── ExaCore ───────────────────────────────────────────────────────────────
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # ── Objective: collect all objective variable indices, single @add_obj ────
    _oidxs = Int32[]
    for K in 1:NZm1, J in 1:NYm1, I in 1:NXm1
        push!(_oidxs, Xinn(I,J,K), Yinn(I,J,K), Zinn(I,J,K))
    end
    for K in 1:NZm1, J in 1:NYm1
        push!(_oidxs, YNX(J,K), ZNX(J,K))
    end
    for K in 1:NZm1, I in 1:NXm1
        push!(_oidxs, XNY(I,K), ZNY(I,K))
    end
    for J in 1:NYm1, I in 1:NXm1
        push!(_oidxs, XNZ(I,J), YNZ(I,J))
    end
    if border_in_obj
        for K in 1:NZ, J in 1:NY
            push!(_oidxs, Y0jk(J,K), Z0jk(J,K), YNXpjk(J,K), ZNXpjk(J,K))
        end
        for K in 1:NZ, I in 1:NX
            push!(_oidxs, Xi0k(I,K), Zi0k(I,K), XiNYpk(I,K), ZiNYpk(I,K))
        end
        for J in 1:NY, I in 1:NX
            push!(_oidxs, Xij0(I,J), Yij0(I,J), XijNZp(I,J), YijNZp(I,J))
        end
    end
    obj_data = [(idx=v,) for v in _oidxs]
    ExaModels.@add_obj(c, T(0.5)*x[d.idx]^2 for d in obj_data)

    # ── Constraints (K outer, J middle, I inner: matches CUTEst ordering) ─────
    con_data = [
        let
            s = Int32(1)
            _hxp = I<=NXm1 && !(J==NY && K==NZ)
            _hxm = I>=2    && !(J==NY && K==NZ)
            _hyp = J<=NYm1 && !(I==NX && K==NZ)
            _hym = J>=2    && !(I==NX && K==NZ)
            xp   = _hxp ? Int32(X(I,J,K))   : s;  cxp   = _hxp ? T(1) : T(0)
            xm   = _hxm ? Int32(X(I-1,J,K)) : s;  cxm   = _hxm ? T(1) : T(0)
            yp   = _hyp ? Int32(Y(I,J,K))   : s;  cyp   = _hyp ? T(1) : T(0)
            ym   = _hym ? Int32(Y(I,J-1,K)) : s;  cym   = _hym ? T(1) : T(0)
            _hzp = K<=NZm1 && !(I==NX && J==NY)
            _hzm = K>=2    && !(I==NX && J==NY)
            zp   = _hzp ? Int32(Z(I,J,K))   : s;  czp   = _hzp ? T(1) : T(0)
            zm   = _hzm ? Int32(Z(I,J,K-1)) : s;  czm   = _hzm ? T(1) : T(0)
            ey0   = (I==1)  ? Int32(Y0jk(J,K))   : s;  cey0   = (I==1)  ? T(1) : T(0)
            eyNXp = (I==NX) ? Int32(YNXpjk(J,K)) : s;  ceyNXp = (I==NX) ? T(1) : T(0)
            ez0   = (I==1)  ? Int32(Z0jk(J,K))   : s;  cez0   = (I==1)  ? T(1) : T(0)
            ezNXp = (I==NX) ? Int32(ZNXpjk(J,K)) : s;  cezNXp = (I==NX) ? T(1) : T(0)
            ex0   = (J==1)  ? Int32(Xi0k(I,K))   : s;  cex0   = (J==1)  ? T(1) : T(0)
            exNYp = (J==NY) ? Int32(XiNYpk(I,K)) : s;  cexNYp = (J==NY) ? T(1) : T(0)
            fz0   = (J==1)  ? Int32(Zi0k(I,K))   : s;  cfz0   = (J==1)  ? T(1) : T(0)
            fzNYp = (J==NY) ? Int32(ZiNYpk(I,K)) : s;  cfzNYp = (J==NY) ? T(1) : T(0)
            gx0   = (K==1)  ? Int32(Xij0(I,J))   : s;  cgx0   = (K==1)  ? T(1) : T(0)
            gxNZp = (K==NZ) ? Int32(XijNZp(I,J)) : s;  cgxNZp = (K==NZ) ? T(1) : T(0)
            gy0   = (K==1)  ? Int32(Yij0(I,J))   : s;  cgy0   = (K==1)  ? T(1) : T(0)
            gyNZp = (K==NZ) ? Int32(YijNZp(I,J)) : s;  cgyNZp = (K==NZ) ? T(1) : T(0)
            (xp=xp, cxp=cxp, xm=xm, cxm=cxm,
             yp=yp, cyp=cyp, ym=ym, cym=cym,
             zp=zp, czp=czp, zm=zm, czm=czm,
             ey0=ey0, cey0=cey0, eyNXp=eyNXp, ceyNXp=ceyNXp,
             ez0=ez0, cez0=cez0, ezNXp=ezNXp, cezNXp=cezNXp,
             ex0=ex0, cex0=cex0, exNYp=exNYp, cexNYp=cexNYp,
             fz0=fz0, cfz0=cfz0, fzNYp=fzNYp, cfzNYp=cfzNYp,
             gx0=gx0, cgx0=cgx0, gxNZp=gxNZp, cgxNZp=cgxNZp,
             gy0=gy0, cgy0=cgy0, gyNZp=gyNZp, cgyNZp=cgyNZp)
        end
        for K in 1:NZ for J in 1:NY for I in 1:NX
    ]

    ExaModels.@add_con(c,
        d.cxp*x[d.xp]   - d.cxm*x[d.xm]     +
        d.cyp*x[d.yp]   - d.cym*x[d.ym]     +
        d.czp*x[d.zp]   - d.czm*x[d.zm]     +
        d.cey0*x[d.ey0] + d.ceyNXp*x[d.eyNXp] +
        d.cez0*x[d.ez0] + d.cezNXp*x[d.ezNXp] +
        d.cex0*x[d.ex0] + d.cexNYp*x[d.exNYp] +
        d.cfz0*x[d.fz0] + d.cfzNYp*x[d.fzNYp] +
        d.cgx0*x[d.gx0] + d.cgxNZp*x[d.gxNZp] +
        d.cgy0*x[d.gy0] + d.cgyNZp*x[d.gyNZp] - T(1)
        for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end

function aug3d_model(NX = 20, NY = 20, NZ = 20; T = Float64, backend = nothing, kwargs...)
    _aug3d_make_model(NX, NY, NZ; border_in_obj = false,
        lb_xface = nothing, lb_yface = nothing, lb_zface = nothing,
        T = T, backend = backend, kwargs...)
end

function aug3dc_model(NX = 20, NY = 20, NZ = 20; T = Float64, backend = nothing, kwargs...)
    _aug3d_make_model(NX, NY, NZ; border_in_obj = true,
        lb_xface = nothing, lb_yface = nothing, lb_zface = nothing,
        T = T, backend = backend, kwargs...)
end

function aug3dqp_model(NX = 20, NY = 20, NZ = 20; T = Float64, backend = nothing, kwargs...)
    _aug3d_make_model(NX, NY, NZ; border_in_obj = false,
        lb_xface = 1.0, lb_yface = 1.0, lb_zface = 1.0,
        T = T, backend = backend, kwargs...)
end

function aug3dcqp_model(NX = 20, NY = 20, NZ = 20; T = Float64, backend = nothing, kwargs...)
    _aug3d_make_model(NX, NY, NZ; border_in_obj = true,
        lb_xface = 1.0, lb_yface = 1.0, lb_zface = 1.0,
        T = T, backend = backend, kwargs...)
end
