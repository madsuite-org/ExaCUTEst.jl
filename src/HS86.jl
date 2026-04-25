function hs86_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf, Inf],
                       start = T[0.0, 0.0, 0.0, 0.0, 1.0])
    # OBJ = E·x + D·x³ + x'Cx (full double sum, symmetric C matrix)
    ExaModels.@add_obj(c,
        -15*x[1] - 27*x[2] - 36*x[3] - 18*x[4] - 12*x[5] +
        4*x[1]^3 + 8*x[2]^3 + 10*x[3]^3 + 6*x[4]^3 + 2*x[5]^3 +
        30*x[1]^2 - 40*x[1]*x[2] - 20*x[1]*x[3] + 64*x[1]*x[4] - 20*x[1]*x[5] +
        39*x[2]^2 - 12*x[2]*x[3] - 62*x[2]*x[4] + 64*x[2]*x[5] +
        10*x[3]^2 - 12*x[3]*x[4] - 20*x[3]*x[5] +
        39*x[4]^2 - 40*x[4]*x[5] +
        30*x[5]^2)
    # 10 linear constraints: sum_j A(i,j)*x[j] - B(i) >= 0
    ExaModels.@add_con(c, -16*x[1] + 2*x[2] + x[4] + 40; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, -2*x[2] + 4*x[4] + 2*x[5] + 2; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, T(-3.5)*x[1] + 2*x[3] + T(0.25); lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, -2*x[2] - 4*x[4] - x[5] + 4; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, -9*x[2] - 2*x[3] + x[4] - T(2.8)*x[5] + 4; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, 2*x[1] - 4*x[3] + 1; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, -x[1] - x[2] - x[3] - x[4] - x[5] + 40; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, -x[1] - 2*x[2] - 3*x[3] - 2*x[4] - x[5] + 60; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, x[1] + 2*x[2] + 3*x[3] + 4*x[4] + 5*x[5] - 5; lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, x[1] + x[2] + x[3] + x[4] + x[5] - 1; lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
