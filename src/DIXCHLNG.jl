# DIXCHLNG: Dixon's challenge problem for SQP methods (Dixon 1991)
# 10 free variables, 5 equality constraints
# Objective = sum_{I=1}^7 [100*(x[I+1]-x[I]^2)^2 + (x[I]-1)^2
#             + 90*(x[I+3]-x[I+2]^2)^2 + (x[I+2]-1)^2
#             + 10.1*(x[I+1]-1)^2 + 10.1*(x[I+3]-1)^2
#             + 19.8*((x[I+1]-1)*(x[I+3]-1))^2]
# Constraints: prod(x[1..2k]) = 1 for k=1..5
# Start: x = [-2, -0.5, 3, 1/3, -4, -0.25, 5, 0.2, -6, -1/6]
function dixchlng_model(; T = Float64, backend = nothing, kwargs...)
    start = T[-2, -0.5, 3, 1//3, -4, -0.25, 5, 0.2, -6, -1//6]
    data_I = [(i = i, ip1 = i + 1, ip2 = i + 2, ip3 = i + 3) for i in 1:7]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 10; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_obj(c,
        T(100) * (x[d.ip1] - x[d.i]^2)^2 + (x[d.i] - T(1))^2 +
        T(90) * (x[d.ip3] - x[d.ip2]^2)^2 + (x[d.ip2] - T(1))^2 +
        T(10.1) * (x[d.ip1] - T(1))^2 + T(10.1) * (x[d.ip3] - T(1))^2 +
        T(19.8) * (x[d.ip1] - T(1)) * (x[d.ip3] - T(1))
        for d in data_I)
    ExaModels.@add_con(c, x[1]*x[2] - T(1))
    ExaModels.@add_con(c, x[1]*x[2]*x[3]*x[4] - T(1))
    ExaModels.@add_con(c, x[1]*x[2]*x[3]*x[4]*x[5]*x[6] - T(1))
    ExaModels.@add_con(c, x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8] - T(1))
    ExaModels.@add_con(c, x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9]*x[10] - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
