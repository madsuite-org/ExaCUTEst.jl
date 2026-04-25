# HS111: Chemical equilibrium (10 variables, 3 linear equality constraints)
# Source: Hock & Schittkowski, problem 111
# SIF: OBJ group uses element type OBJ with all 10 vars + parameter C[i].
#   Each element O(I) contributes: exp(x[i])*(C[i] + x[i] - log(sum_j exp(x[j])))
#   Constraints: purely linear E-type with RHS constants subtracted.
const C_HS111 = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]

function hs111_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 10;
        lvar  = T(-100.0),
        uvar  = T(100.0),
        start = T(-2.3))
    ExaModels.@add_par(c, C, C_HS111)
    ExaModels.@add_obj(c,
        exp(x[i]) * (C[i] + x[i] - log(
            exp(x[1]) + exp(x[2]) + exp(x[3]) + exp(x[4]) + exp(x[5]) +
            exp(x[6]) + exp(x[7]) + exp(x[8]) + exp(x[9]) + exp(x[10])
        )) for i in 1:10)
    # Equality constraints (E-type), CUTEst residual form: g(x) - rhs = 0
    ExaModels.@add_con(c, exp(x[1]) + T(2)*exp(x[2]) + T(2)*exp(x[3]) + exp(x[6]) + exp(x[10]) - T(2))
    ExaModels.@add_con(c, exp(x[4]) + T(2)*exp(x[5]) + exp(x[6]) + exp(x[7]) - T(1))
    ExaModels.@add_con(c, exp(x[3]) + exp(x[7]) + exp(x[8]) + T(2)*exp(x[9]) + exp(x[10]) - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
