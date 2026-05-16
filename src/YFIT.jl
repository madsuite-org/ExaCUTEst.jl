function _yfit_core(; T = Float64, backend = nothing)
    # Shared data for YFIT, YFITNE, YFITU
    # tanab element: d1 * tan(a1*(1 - i/p) + b1*(i/p))  for i=0..p, p=16
    nothing
end

function yfit_model(; T = Float64, backend = nothing, kwargs...)
    # YFIT: least squares fit with tan model, bounds: alpha,beta free; dist≥0
    # Objective: sum_{i=0}^{16} (dist*tan(alpha*(1-i/16) + beta*(i/16)) - y_i)^2

    p = 16
    y_data = T[21.158931, 17.591719, 14.046854, 10.519732, 7.0058392, 3.5007293,
               0.0000000, -3.5007293, -7.0058392, -10.519732, -14.046854, -17.591719,
               -21.158931, -24.753206, -28.379405, -32.042552, -35.747869]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, 0.0],
        uvar = T[Inf, Inf, Inf],
        start = T[0.60, -0.60, 20.0])

    # x[1]=alpha, x[2]=beta, x[3]=dist
    ExaModels.@add_par(c, y_par, y_data)
    ExaModels.@add_par(c, frac_par, T[T(i) / T(p) for i in 0:p])
    # GROUP TYPE square: F(v)=v^2, group diff(i): element tanab - y_i
    ExaModels.@add_obj(c,
        (x[3] * tan(x[1] * (T(1) - frac_par[i]) + x[2] * frac_par[i]) - y_par[i])^2
        for i in 1:(p+1))

    return ExaModels.ExaModel(c; kwargs...)
end

function yfitu_model(; T = Float64, backend = nothing, kwargs...)
    # YFITU: unconstrained variant of YFIT (all variables free)

    p = 16
    y_data = T[21.158931, 17.591719, 14.046854, 10.519732, 7.0058392, 3.5007293,
               0.0000000, -3.5007293, -7.0058392, -10.519732, -14.046854, -17.591719,
               -21.158931, -24.753206, -28.379405, -32.042552, -35.747869]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf],
        start = T[0.60, -0.60, 20.0])

    ExaModels.@add_par(c, y_par, y_data)
    ExaModels.@add_par(c, frac_par, T[T(i) / T(p) for i in 0:p])
    ExaModels.@add_obj(c,
        (x[3] * tan(x[1] * (T(1) - frac_par[i]) + x[2] * frac_par[i]) - y_par[i])^2
        for i in 1:(p+1))

    return ExaModels.ExaModel(c; kwargs...)
end

function yfitne_model(; T = Float64, backend = nothing, kwargs...)
    # YFITNE: nonlinear equation variant of YFIT, all free variables
    # 17 equality constraints: dist*tan(alpha*(1-i/16)+beta*(i/16)) = y_i for i=0..16

    p = 16
    y_data = T[21.158931, 17.591719, 14.046854, 10.519732, 7.0058392, 3.5007293,
               0.0000000, -3.5007293, -7.0058392, -10.519732, -14.046854, -17.591719,
               -21.158931, -24.753206, -28.379405, -32.042552, -35.747869]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
        lvar = T[-Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, Inf],
        start = T[0.60, -0.60, 20.0])

    ExaModels.@add_par(c, y_par, y_data)
    ExaModels.@add_par(c, frac_par, T[T(i) / T(p) for i in 0:p])
    # E-type constraints (equality): element - y_i = 0
    ExaModels.@add_con(c,
        x[3] * tan(x[1] * (T(1) - frac_par[i]) + x[2] * frac_par[i]) - y_par[i]
        for i in 1:(p+1))

    return ExaModels.ExaModel(c; kwargs...)
end
