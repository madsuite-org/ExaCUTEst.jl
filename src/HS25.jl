const U_HS25 = [(-50 * log(i / 100))^(2 / 3) + 25 for i in 1:99]
const C_HS25 = [i / 100 for i in 1:99]

function hs25_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       lvar  = T[0.1, 0.0, 0.0],
                       uvar  = T[100.0, 25.6, 5.0],
                       start = T[100.0, 12.5, 3.0])
    ExaModels.@add_par(c, U, U_HS25)
    ExaModels.@add_par(c, C, C_HS25)
    # 99 L2 groups: WFI element = exp(-(U[i]-x2)^x3 / x1) (SIF: XI=1/X, F=exp(-XI*WMY^Z))
    # contribution = (exp(-(U[i]-x2)^x3 / x1) - C[i])^2
    ExaModels.@add_obj(c,
        (exp(-(U[i] - x[2])^x[3] / x[1]) - C[i])^2 for i in 1:99)
    return ExaModels.ExaModel(c; kwargs...)
end
