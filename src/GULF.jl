function gulf_model(; T = Float64, backend = nothing, kwargs...)
    # GULF: Gulf Research and Development function (3 variables, unconstrained)
    # 99 L2 groups, each with element GLF(X1,X2,X3; T_i)
    # GLF element: F = exp(-A), A = |YMV2|^X3 / X1, YMV2 = 25 + (-50*log(T_i))^(2/3) - X2
    # Group i: v_i = exp(-A_i) - T_i, contribution = v_i^2  (L2, no scale)
    # BOUNDS: FR DEFAULT → all variables free
    # Note: at start point YMV2 > 0 for all i, so abs(YMV2) = YMV2.
    #
    # KNOWN CUTEST HESSIAN BUG: The GULF.SIF element GLF contains incorrect formulas for
    # H[V1,V3] and H[V2,V3]. The SIF computes d(A·EXPMA)/dV3 by treating A as a constant
    # multiplier (only differentiating through EXPMA), giving -A²·log(y)·exp(-A)/V1 instead
    # of the correct A·(1-A)·log(y)·exp(-A)/V1. This is provably inconsistent with the
    # gradient formulas also specified in the SIF. ExaModels computes the mathematically
    # correct Hessian (verified by finite differences); the Hessian test for GULF uses
    # @test_broken to document this CUTEst discrepancy.
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf], start = T[5.0, 2.5, 0.15])
    # Parameter: TI[i] = i * 0.01
    ExaModels.@add_par(c, TI, T[i * 0.01 for i in 1:99])
    # Precompute YB[i] = 25 + (-50*log(TI[i]))^(2/3)  (constant part of YMV2)
    ExaModels.@add_par(c, YB, T[25.0 + (-50.0 * log(i * 0.01))^(2.0/3.0) for i in 1:99])
    # A = (YB[i] - x2)^x3 / x1  (YMV2 = YB[i]-x2 > 0 near start point)
    # v_i = exp(-A) - TI[i], contribution = v_i^2
    ExaModels.@add_obj(c, (exp(-(YB[i] - x[2])^x[3] / x[1]) - TI[i])^2 for i in 1:99)
    return ExaModels.ExaModel(c; kwargs...)
end
