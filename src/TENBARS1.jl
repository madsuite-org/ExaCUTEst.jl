function _tenbars_model(; T = Float64, backend = nothing,
                         stress_type = :none,
                         u_lower_bounds = Int[],  # indices with lvar=-50.8
                         kwargs...)
    # Ten-bar truss: 18 variables U(1..8) [x[1..8]], X(1..10) [x[9..18]]
    # Elements (expand IV to inline):
    #   EA  = X1*U1              (XU)
    #   EB  = X3*(U1+U2)         (XUPV)
    #   EC  = X6*(U1-U5)         (XUMV)
    #   ED  = X7*(U1+U8-U2-U7)   (XBIG)
    #   EE  = X5*(U2-U4)         (XUMV)
    #   EF  = X2*(U3-U4)         (XUMV — default type, no T card in SIF)
    #   EG  = X4*U3              (XU)
    #   EH  = X8*(U3+U4-U5-U6)   (XBIG)
    #   EI  = X9*(U3-U7)         (XUMV)
    #   EJ  = X10*(U6-U8)        (XUMV)

    C0   = 2.53106
    SQ2  = sqrt(2.0)
    ISQ8 = 1.0 / sqrt(8.0)
    P0   = 589.884  # magnitude (SIF has -P0 = -589.884, RHS sign reversal gives +P0)

    lvar = fill(T(-Inf), 18)
    for i in u_lower_bounds; lvar[i] = T(-50.8); end
    for j in 1:10; lvar[8+j] = T(0.645); end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 18; lvar = lvar, uvar = T(Inf), start = T(0))

    # Objective: C0*(X1+X4+X5+X6+X9+X10) + C0*SQ2*(X2+X3+X7+X8)
    ExaModels.@add_obj(c,
        T(C0)    * (x[9]  + x[12] + x[13] + x[14] + x[17] + x[18])
      + T(C0*SQ2)* (x[10] + x[11] + x[15] + x[16]))

    # Purely linear constraint comes first in CUTEst (sifdecoder reorders).
    # When a linear constraint is present: ordering is linear, C2,C3,...,C8,C1 (C1 last).
    # When no linear constraint: ordering is C1,C2,C3,...,C8 (natural SIF order).
    has_linear = stress_type != :none

    if stress_type == :stress
        ExaModels.@add_con(c, x[2] - x[4] - T(22.86); lcon = T(-Inf), ucon = T(0))
    elseif stress_type == :strain
        ExaModels.@add_con(c, x[4] + x[8] + T(76.2); lcon = T(0), ucon = T(Inf))
    end

    # C1: EA + ISQ8*EB + EC + ISQ8*ED = 0
    if !has_linear
        ExaModels.@add_con(c,
            x[9]*x[1]
          + T(ISQ8)*x[11]*(x[1]+x[2])
          + x[14]*(x[1]-x[5])
          + T(ISQ8)*x[15]*(x[1]+x[8]-x[2]-x[7]))
    end

    # C2: ISQ8*EB + EE - ISQ8*ED = 0
    ExaModels.@add_con(c,
        T(ISQ8)*x[11]*(x[1]+x[2])
      + x[13]*(x[2]-x[4])
      - T(ISQ8)*x[15]*(x[1]+x[8]-x[2]-x[7]))

    # C3: ISQ8*EF + ISQ8*EH + EI + EG = 0  (EF = X2*(U3-U4), XUMV type)
    ExaModels.@add_con(c,
        T(ISQ8)*x[10]*(x[3]-x[4])
      + T(ISQ8)*x[16]*(x[3]+x[4]-x[5]-x[6])
      + x[17]*(x[3]-x[7])
      + x[12]*x[3])

    # C4: -ISQ8*EF + ISQ8*EH - EE + P0 = 0  (EF = X2*(U3-U4))
    ExaModels.@add_con(c,
      - T(ISQ8)*x[10]*(x[3]-x[4])
      + T(ISQ8)*x[16]*(x[3]+x[4]-x[5]-x[6])
      - x[13]*(x[2]-x[4])
      + T(P0))

    # C5: -ISQ8*EH - EC = 0
    ExaModels.@add_con(c,
      - T(ISQ8)*x[16]*(x[3]+x[4]-x[5]-x[6])
      - x[14]*(x[1]-x[5]))

    # C6: -ISQ8*EH + EJ = 0
    ExaModels.@add_con(c,
      - T(ISQ8)*x[16]*(x[3]+x[4]-x[5]-x[6])
      + x[18]*(x[6]-x[8]))

    # C7: -ISQ8*ED - EI = 0
    ExaModels.@add_con(c,
      - T(ISQ8)*x[15]*(x[1]+x[8]-x[2]-x[7])
      - x[17]*(x[3]-x[7]))

    # C8: ISQ8*ED - EJ + P0 = 0
    ExaModels.@add_con(c,
        T(ISQ8)*x[15]*(x[1]+x[8]-x[2]-x[7])
      - x[18]*(x[6]-x[8])
      + T(P0))

    # C1 goes last when a linear constraint is present
    if has_linear
        ExaModels.@add_con(c,
            x[9]*x[1]
          + T(ISQ8)*x[11]*(x[1]+x[2])
          + x[14]*(x[1]-x[5])
          + T(ISQ8)*x[15]*(x[1]+x[8]-x[2]-x[7]))
    end

    return ExaModels.ExaModel(c; kwargs...)
end

tenbars1_model(; T = Float64, backend = nothing, kwargs...) =
    _tenbars_model(; T=T, backend=backend, stress_type=:stress,
                   u_lower_bounds=[2,4,6,8], kwargs...)

tenbars2_model(; T = Float64, backend = nothing, kwargs...) =
    _tenbars_model(; T=T, backend=backend, stress_type=:none,
                   u_lower_bounds=[2,4,6,8], kwargs...)

tenbars3_model(; T = Float64, backend = nothing, kwargs...) =
    _tenbars_model(; T=T, backend=backend, stress_type=:none,
                   u_lower_bounds=[4,8], kwargs...)

tenbars4_model(; T = Float64, backend = nothing, kwargs...) =
    _tenbars_model(; T=T, backend=backend, stress_type=:strain,
                   u_lower_bounds=[2,4,6,8], kwargs...)
