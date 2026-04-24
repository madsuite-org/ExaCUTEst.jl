function hs56_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7;
                       lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[1.0, 1.0, 1.0, 0.50973968, 0.50973968, 0.50973968, 0.98511078])
    # OBJ = -x1*x2*x3  (3PROD element with coeff -1)
    ExaModels.@add_obj(c, -x[1]*x[2]*x[3])
    # PSNSQ(V1,P) = -P*sin(V1)^2; PAR = [4.2, 4.2, 4.2, 7.2]
    ExaModels.@add_con(c, x[1] - T(4.2)*sin(x[4])^2)
    ExaModels.@add_con(c, x[2] - T(4.2)*sin(x[5])^2)
    ExaModels.@add_con(c, x[3] - T(4.2)*sin(x[6])^2)
    ExaModels.@add_con(c, x[1] + 2*x[2] + 2*x[3] - T(7.2)*sin(x[7])^2)
    return ExaModels.ExaModel(c; kwargs...)
end
