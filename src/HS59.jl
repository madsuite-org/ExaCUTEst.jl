function hs59_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # No FR: default lvar=0; UP bounds given
    ExaModels.@add_var(c, x, 2; lvar = T[0.0, 0.0], uvar = T[75.0, 65.0],
                       start = T[90.0, 10.0])
    # OBJ: complex polynomial + exp + rational term; constant RHS=75.196
    ExaModels.@add_obj(c,
        T(3.8112)*x[1] + T(6.8306)*x[2]
        + T(0.0020567)*x[1]^3 - T(1.0345e-5)*x[1]^4
        - T(0.030234)*x[1]*x[2] + T(1.28134e-3)*x[1]^2*x[2]
        + T(2.266e-7)*x[1]^4*x[2] - T(0.25645)*x[2]^2
        + T(0.0034604)*x[2]^3 - T(1.3514e-5)*x[2]^4
        + T(28.106)/(x[2] + 1)
        + T(5.2375e-6)*x[1]^2*x[2]^2 + T(6.3e-8)*x[1]^3*x[2]^2
        - T(7.0e-10)*x[1]^3*x[2]^3 - T(3.405e-4)*x[1]*x[2]^2
        + T(1.6638e-6)*x[1]*x[2]^3
        + T(2.8673)*exp(T(0.0005)*x[1]*x[2])
        - T(3.5256e-5)*x[1]^3*x[2] - T(0.12694)*x[1]^2
        - T(75.196))
    # CON1 (G): x1*x2 - 700 >= 0
    ExaModels.@add_con(c, x[1]*x[2] - 700; lcon = T(0), ucon = T(Inf))
    # CON2 (G): x2 - 0.008*x1^2 >= 0
    ExaModels.@add_con(c, x[2] - T(0.008)*x[1]^2; lcon = T(0), ucon = T(Inf))
    # CON3 (G): -5*x1 + (x2-50)^2 + 275 >= 0
    ExaModels.@add_con(c, -5*x[1] + (x[2] - 50)^2 + 275; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
