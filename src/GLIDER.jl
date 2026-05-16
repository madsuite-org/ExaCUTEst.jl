# GLIDER: Hang glider trajectory optimization (COPS problem 11)
# Maximize final horizontal position in presence of a thermal updraft.
# NH=400 subintervals; state (X,Y,VX,VY), control CL, and auxiliary R,U,W,V,D,L,VX.,VY.
# Variables: STEP, then for I=0..NH: X(I),Y(I),VX(I),VY(I),CL(I),R(I),U(I),W(I),V(I),D(I),L(I),VX.(I),VY.(I)
# Source: E. Dolan and J. More', "Benchmarking Optimization Software with COPS", 2000
# SIF input: Nick Gould, December 2000
function glider_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    # Physical parameters
    X0_f   = 0.0; Y0_f  = 1000.0; VX0_f = 13.23; VY0_f = -1.288
    YF_f   = 900.0; VXF_f = 13.23; VYF_f = -1.288
    UC_f   = 2.5; R0_f  = 100.0; C0_f  = 0.034; C1_f  = 0.069662
    M_f    = 100.0; S_f  = 14.0; RHO_f = 1.13; G_f   = 9.81
    CLMIN_f = 0.0; CLMAX_f = 1.4

    NH_f   = Float64(NH)
    H_f    = 1.0 / NH_f            # step size (STEP variable ≈ 1/NH at start)
    R02_f  = R0_f^2
    RHOS2_f = RHO_f * S_f * 0.5   # rho*S/2
    INV1M_f = 1.0 / M_f

    X0 = T(X0_f); Y0 = T(Y0_f); VX0 = T(VX0_f); VY0 = T(VY0_f)
    YF = T(YF_f); VXF = T(VXF_f); VYF = T(VYF_f)
    UC = T(UC_f); G = T(G_f)
    C0 = T(C0_f); C1 = T(C1_f)
    CLMIN = T(CLMIN_f); CLMAX = T(CLMAX_f)
    CLI = CLMAX * T(0.5)

    # Variable indexing (1-based Julia)
    # 1: STEP
    # For I=0..NH (NH+1 nodes, 13 variables each):
    #   base(I) = 2 + I*13
    #   X(I)   = base(I) + 1
    #   Y(I)   = base(I) + 2
    #   VX(I)  = base(I) + 3
    #   VY(I)  = base(I) + 4
    #   CL(I)  = base(I) + 5
    #   R(I)   = base(I) + 6
    #   U(I)   = base(I) + 7
    #   W(I)   = base(I) + 8
    #   V(I)   = base(I) + 9
    #   D(I)   = base(I) + 10
    #   L(I)   = base(I) + 11
    #   VX.(I) = base(I) + 12
    #   VY.(I) = base(I) + 13
    STEP_IDX = 1
    per_node = 13
    base(I)  = 2 + I * per_node   # I is 0-based
    xi(I)    = base(I) + 1
    yi(I)    = base(I) + 2
    vxi(I)   = base(I) + 3
    vyi(I)   = base(I) + 4
    cli(I)   = base(I) + 5
    ri(I)    = base(I) + 6
    ui(I)    = base(I) + 7
    wi(I)    = base(I) + 8
    vi(I)    = base(I) + 9
    di(I)    = base(I) + 10
    li(I)    = base(I) + 11
    vxd(I)   = base(I) + 12
    vyd(I)   = base(I) + 13

    NV = 1 + (NH + 1) * per_node

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    x0v  = zeros(T, NV)

    # STEP >= 0, start = 1/NH
    lvar[STEP_IDX] = T(0)
    x0v[STEP_IDX]  = T(H_f)

    # Per node bounds
    for I in 0:NH
        # CL in [CLMIN, CLMAX]
        lvar[cli(I)] = CLMIN
        uvar[cli(I)] = CLMAX
        # X >= 0
        lvar[xi(I)] = T(0)
        # VX >= 0
        lvar[vxi(I)] = T(0)
        # Initial starting point
        x0v[xi(I)]   = T(X0_f + I * VX0_f / NH_f)
        x0v[yi(I)]   = T(Y0_f + I * (YF_f - Y0_f) / NH_f)
        x0v[vxi(I)]  = T(VX0_f)
        x0v[vyi(I)]  = T(VY0_f)
        x0v[cli(I)]  = CLI
        # Compute auxiliary starting points
        rI  = (x0v[xi(I)] / T(R0_f) - T(2.5))^2
        expR = exp(-rI)
        uI  = T(UC_f) * (T(1) - rI) * expR
        wI  = T(VY0_f) - uI
        vI  = sqrt(T(VX0_f)^2 + wI^2)
        dI  = T(RHOS2_f) * (T(C0_f) + T(C1_f) * CLI^2) * vI^2
        lI  = T(RHOS2_f) * CLI * vI^2
        vxdI = (-lI * (wI/vI) - dI * (T(VX0_f)/vI)) / T(M_f)
        vydI = (lI * (T(VX0_f)/vI) - dI * (wI/vI)) / T(M_f) - T(G_f)
        x0v[ri(I)]  = rI
        x0v[ui(I)]  = uI
        x0v[wi(I)]  = wI
        x0v[vi(I)]  = vI
        x0v[di(I)]  = dI
        x0v[li(I)]  = lI
        x0v[vxd(I)] = vxdI
        x0v[vyd(I)] = vydI
    end

    # Fixed boundary conditions
    lvar[xi(0)]   = X0;  uvar[xi(0)]  = X0
    lvar[yi(0)]   = Y0;  uvar[yi(0)]  = Y0
    lvar[vxi(0)]  = VX0; uvar[vxi(0)] = VX0
    lvar[vyi(0)]  = VY0; uvar[vyi(0)] = VY0
    lvar[yi(NH)]  = YF;  uvar[yi(NH)] = YF
    lvar[vxi(NH)] = VXF; uvar[vxi(NH)] = VXF
    lvar[vyi(NH)] = VYF; uvar[vyi(NH)] = VYF

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0v)

    # === Objective: minimize -X(NH) ===
    ExaModels.@add_obj(c, -x[xi(NH)])

    # === Trapezoidal constraints for X, Y, VX, VY ===
    # X(J) - X(J-1) - 0.5*STEP*(VX(J) + VX(J-1)) = 0   for J=1..NH
    # Y(J) - Y(J-1) - 0.5*STEP*(VY(J) + VY(J-1)) = 0
    # VX(J) - VX(J-1) - 0.5*STEP*(VX.(J) + VX.(J-1)) = 0
    # VY(J) - VY(J-1) - 0.5*STEP*(VY.(J) + VY.(J-1)) = 0
    trap_data = [(
        xJ   = xi(J),  xJm1  = xi(J-1),
        yJ   = yi(J),  yJm1  = yi(J-1),
        vxJ  = vxi(J), vxJm1 = vxi(J-1),
        vyJ  = vyi(J), vyJm1 = vyi(J-1),
        vxdJ = vxd(J), vxdJm1 = vxd(J-1),
        vydJ = vyd(J), vydJm1 = vyd(J-1),
    ) for J in 1:NH]

    ExaModels.@add_con(c,
        x[d.xJ] - x[d.xJm1] - T(0.5)*x[STEP_IDX]*(x[d.vxJ] + x[d.vxJm1])
        for d in trap_data)

    ExaModels.@add_con(c,
        x[d.yJ] - x[d.yJm1] - T(0.5)*x[STEP_IDX]*(x[d.vyJ] + x[d.vyJm1])
        for d in trap_data)

    ExaModels.@add_con(c,
        x[d.vxJ] - x[d.vxJm1] - T(0.5)*x[STEP_IDX]*(x[d.vxdJ] + x[d.vxdJm1])
        for d in trap_data)

    ExaModels.@add_con(c,
        x[d.vyJ] - x[d.vyJm1] - T(0.5)*x[STEP_IDX]*(x[d.vydJ] + x[d.vydJm1])
        for d in trap_data)

    # === Auxiliary variable definition constraints for I=0..NH ===
    # R(I) = (X(I)/R0 - 2.5)^2  => R(I) - (X(I)/R0 - 2.5)^2 = 0
    # Encoding: -R(I) + (X(I) - 2.5*R0)^2 / R0^2 = 0
    INV_R02 = T(1.0 / R02_f)
    SHIFT   = T(2.5 * R0_f)

    aux_data = [(
        xi_  = xi(I),  ri_ = ri(I),  ui_ = ui(I),
        wi_  = wi(I),  vi_ = vi(I),  di_ = di(I),
        li_  = li(I),  vxi_ = vxi(I), vyi_ = vyi(I),
        vxdi = vxd(I), vydi = vyd(I),
    ) for I in 0:NH]

    # R(I) + (X(I) - SHIFT)^2 / R02 = 0 but with linear: -R(I) + ...
    ExaModels.@add_con(c,
        -x[d.ri_] + INV_R02 * (x[d.xi_] - SHIFT)^2
        for d in aux_data)

    # U(I) = UC*(1-R(I))*exp(-R(I))  => -U(I) + UC*(1-R(I))*exp(-R(I)) = 0
    ExaModels.@add_con(c,
        -x[d.ui_] + UC * (T(1) - x[d.ri_]) * exp(-x[d.ri_])
        for d in aux_data)

    # W(I) = VY(I) - U(I)  => linear: W(I) - VY(I) + U(I) = 0
    ExaModels.@add_con(c,
        x[d.wi_] - x[d.vyi_] + x[d.ui_]
        for d in aux_data)

    # V(I) = sqrt(VX(I)^2 + W(I)^2)  => -V(I) + sqrt(VX(I)^2 + W(I)^2) = 0
    ExaModels.@add_con(c,
        -x[d.vi_] + sqrt(x[d.vxi_]^2 + x[d.wi_]^2)
        for d in aux_data)

    # D(I) = 0.5*(C0+C1*CL(I)^2)*rho*S*V(I)^2
    # => -D(I) + RHOS2*(C0*V^2 + C1*CL^2*V^2) = 0
    RHOS2 = T(RHOS2_f)
    cli_data = [(
        di_ = di(I), vi_ = vi(I), cli_ = cli(I),
    ) for I in 0:NH]

    ExaModels.@add_con(c,
        -x[d.di_] + RHOS2 * (C0 * x[d.vi_]^2 + C1 * x[d.cli_]^2 * x[d.vi_]^2)
        for d in cli_data)

    # L(I) = 0.5*CL(I)*rho*S*V(I)^2  => -L(I) + RHOS2*CL(I)*V(I)^2 = 0
    ExaModels.@add_con(c,
        -x[d.li_] + RHOS2 * x[d.cli_] * x[d.vi_]^2
        for d in cli_data)

    # VX.(I) = (-L(I)*W(I)/V(I) - D(I)*VX(I)/V(I)) / M
    # => -VX.(I) + (1/M)*(-L*W/V - D*VX/V) = 0
    # => -VX.(I) - (1/M)*( L(I)*W(I) + D(I)*VX(I) ) / V(I) = 0
    INV_M = T(INV1M_f)
    lv_data = [(
        vxdi_ = vxd(I), vydi_ = vyd(I),
        li_   = li(I),  di_   = di(I),
        wi_   = wi(I),  vxi_  = vxi(I),
        vi_   = vi(I),
    ) for I in 0:NH]

    ExaModels.@add_con(c,
        -x[d.vxdi_] - INV_M * (x[d.li_]*x[d.wi_] + x[d.di_]*x[d.vxi_]) / x[d.vi_]
        for d in lv_data)

    # VY.(I) = (L(I)*VX(I)/V(I) - D(I)*W(I)/V(I)) / M - G
    # => -VY.(I) + (1/M)*(L(I)*VX(I) - D(I)*W(I)) / V(I) - G = 0
    # With CONSTANT G on group: group value = -VY.(I) + nonlinear - G
    # CUTEst: Z GLIDER VY.(I) G => constant G subtracted from group RHS
    # So constraint: -VY.(I) + nonlinear - G = 0 (lcon=ucon=0)
    ExaModels.@add_con(c,
        -x[d.vydi_] + INV_M * (x[d.li_]*x[d.vxi_] - x[d.di_]*x[d.wi_]) / x[d.vi_] - G
        for d in lv_data)

    return ExaModels.ExaModel(c; kwargs...)
end
