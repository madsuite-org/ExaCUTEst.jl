# HS111LNP: HS111 with inactive bound constraints removed (Toint/Plantenga 1992)
# Same as HS111 but all variables free (no bounds); start = -2.3
const C_HS111LNP = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]

function hs111lnp_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 10; lvar = T(-Inf), uvar = T(Inf), start = T(-2.3))
    ExaModels.@add_par(c, C, C_HS111LNP)
    ExaModels.@add_obj(c,
        exp(x[i]) * (C[i] + x[i] - log(
            exp(x[1]) + exp(x[2]) + exp(x[3]) + exp(x[4]) + exp(x[5]) +
            exp(x[6]) + exp(x[7]) + exp(x[8]) + exp(x[9]) + exp(x[10])
        )) for i in 1:10)
    ExaModels.@add_con(c, exp(x[1]) + T(2)*exp(x[2]) + T(2)*exp(x[3]) + exp(x[6]) + exp(x[10]) - T(2))
    ExaModels.@add_con(c, exp(x[4]) + T(2)*exp(x[5]) + exp(x[6]) + exp(x[7]) - T(1))
    ExaModels.@add_con(c, exp(x[3]) + exp(x[7]) + exp(x[8]) + T(2)*exp(x[9]) + exp(x[10]) - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
