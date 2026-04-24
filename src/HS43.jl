function hs43_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[-Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf],
                       start = T[0.0, 0.0, 0.0, 0.0])
    # OBJ: E1+E2+E3+E4 = 1*x1² + 1*x2² + 2*x3² + 1*x4² (PSQ element with P=FAC3=2)
    ExaModels.@add_obj(c,
        x[1]^2 + x[2]^2 + T(2)*x[3]^2 + x[4]^2 -
        T(5)*x[1] - T(5)*x[2] - T(21)*x[3] + T(7)*x[4])
    # CON1 (G): -x1+x2-x3+x4 - x1^2-x2^2-x3^2-x4^2 + 8 ≥ 0
    ExaModels.@add_con(c,
        -x[1] + x[2] - x[3] + x[4] -
        x[1]^2 - x[2]^2 - x[3]^2 - x[4]^2 + T(8);
        lcon = T(0), ucon = T(Inf))
    # CON2 (G): x1+x4 - x1^2-2x2^2-x3^2-2x4^2 + 10 ≥ 0
    ExaModels.@add_con(c,
        x[1] + x[4] -
        x[1]^2 - T(2)*x[2]^2 - x[3]^2 - T(2)*x[4]^2 + T(10);
        lcon = T(0), ucon = T(Inf))
    # CON3 (G): -2x1+x2+x4 - 2x1^2-x2^2-x3^2 + 5 ≥ 0
    ExaModels.@add_con(c,
        -T(2)*x[1] + x[2] + x[4] -
        T(2)*x[1]^2 - x[2]^2 - x[3]^2 + T(5);
        lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
