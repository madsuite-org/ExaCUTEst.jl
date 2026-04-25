function hs93_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[5.54, 4.4, 12.02, 11.82, 0.702, 0.852])
    ExaModels.@add_obj(c, T(2.04e-2)*x[1]*x[4]*(x[1]+x[2]+x[3]) +
                          T(1.87e-2)*x[2]*x[3]*(x[1]+T(1.57)*x[2]+x[4]) +
                          T(6.07e-2)*x[1]*x[4]*x[5]^2*(x[1]+x[2]+x[3]) +
                          T(4.37e-2)*x[2]*x[3]*x[6]^2*(x[1]+T(1.57)*x[2]+x[4]))
    ExaModels.@add_con(c, T(1.0e-3)*x[1]*x[2]*x[3]*x[4]*x[5]*x[6] - T(2.07);
                       lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, T(6.2e-4)*x[1]*x[4]*x[5]^2*(x[1]+x[2]+x[3]) +
                          T(5.8e-4)*x[2]*x[3]*x[6]^2*(x[1]+T(1.57)*x[2]+x[4]) - T(1.0);
                       lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
