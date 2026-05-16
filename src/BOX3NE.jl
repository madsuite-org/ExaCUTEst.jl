# BOX3NE: Box problem in 3 variables, nonlinear equations version
# 3 variables, M=10 equality constraints
# G(i): exp(-0.1*i*x[1]) - exp(-0.1*i*x[2]) + C[i]*x[3] = 0
# where C[i] = exp(-i) - exp(-0.1*i)
# All variables free; start: x = [0, 10, 1]
function box3ne_model(M = 10; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3;
                       start = T[0.0, 10.0, 1.0])

    ExaModels.@add_par(c, TI, T[-0.1 * i for i in 1:M])
    ExaModels.@add_par(c, C, T[exp(-float(i)) - exp(-0.1 * i) for i in 1:M])

    ExaModels.@add_con(c,
        exp(TI[i] * x[1]) - exp(TI[i] * x[2]) + C[i] * x[3]
        for i in 1:M)

    return ExaModels.ExaModel(c; kwargs...)
end
