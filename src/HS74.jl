function hs74_model(; T = Float64, backend = nothing, kwargs...)
    # A = 0.55
    # CUTEst ordering: x[1]=X4, x[2]=X3, x[3]=X2, x[4]=X1
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
                       lvar = T[-0.55, -0.55, 0.0, 0.0],
                       uvar = T[0.55, 0.55, 1200.0, 1200.0],
                       start = T[0.0, 0.0, 0.0, 0.0])
    ExaModels.@add_obj(c, 3*x[4] + 2*x[3] + T(1e-6)*x[4]^3 + T(2e-6/3)*x[3]^3)
    # C1 (G): X4-X3+0.55 = x[1]-x[2]+0.55 ≥ 0
    ExaModels.@add_con(c, x[1]-x[2]+T(0.55); lcon=T(0), ucon=T(Inf))
    # C2 (G): X3-X4+0.55 = x[2]-x[1]+0.55 ≥ 0
    ExaModels.@add_con(c, x[2]-x[1]+T(0.55); lcon=T(0), ucon=T(Inf))
    # C3 (E): -X1-1000*sin(X3+0.25)-1000*sin(X4+0.25)+894.8 = 0
    ExaModels.@add_con(c, -x[4]-1000*sin(x[2]+T(0.25))-1000*sin(x[1]+T(0.25))+T(894.8))
    # C4 (E): -X2+1000*sin(X3-0.25)+1000*sin(X3-X4-0.25)+894.8 = 0
    ExaModels.@add_con(c, -x[3]+1000*sin(x[2]-T(0.25))+1000*sin(x[2]-x[1]-T(0.25))+T(894.8))
    # C5 (E): 1000*sin(X4-0.25)+1000*sin(X4-X3-0.25)+1294.8 = 0
    ExaModels.@add_con(c, 1000*sin(x[1]-T(0.25))+1000*sin(x[1]-x[2]-T(0.25))+T(1294.8))
    return ExaModels.ExaModel(c; kwargs...)
end
