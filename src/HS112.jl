# HS112: Chemical equilibrium (10 variables, 3 linear equality constraints)
# Source: Hock & Schittkowski, problem 112 (= problem 80 in original HS book)
# SIF: objective uses LOG element (x[i]*log(x[i])) and LOGSUM element (-x[i]*log(sum_j x[j]))
#   plus linear C(I)*x[i]; constraints are purely linear with constants subtracted.
#   OBJ = sum_i C[i]*x[i] + x[i]*log(x[i]) - x[i]*log(S)   where S = sum_j x[j]
#   CON1: x1+2x2+2x3+x6+x10 - 2 = 0
#   CON2: x4+2x5+x6+x7 - 1 = 0
#   CON3: x3+x7+x8+2x9+x10 - 1 = 0
const C_HS112 = [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179]

function hs112_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 10;
        lvar  = T[1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        uvar  = T[Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf,  Inf ],
        start = T[0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1,  0.1])
    ExaModels.@add_par(c, C, C_HS112)
    # Linear part: sum C[i]*x[i]
    ExaModels.@add_obj(c, C[i] * x[i] for i in 1:10)
    # LOG element: x[i]*log(x[i]) and LOGSUM element: -x[i]*log(sum_j x[j])
    ExaModels.@add_obj(c, x[i] * log(x[i]) - x[i] * log(x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]) for i in 1:10)
    # Equality constraints (purely linear, E type, constant subtracted)
    ExaModels.@add_con(c, x[1] + T(2)*x[2] + T(2)*x[3] + x[6] + x[10] - T(2))
    ExaModels.@add_con(c, x[4] + T(2)*x[5] + x[6] + x[7] - T(1))
    ExaModels.@add_con(c, x[3] + x[7] + x[8] + T(2)*x[9] + x[10] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
