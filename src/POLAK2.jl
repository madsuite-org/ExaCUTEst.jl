# POLAK2: Nonlinear minmax in ten variables (Polak, Mayne & Higgins 1991)
# 11 free variables: X(1..10), U; start X1=100, rest=0.1
# Objective: U; 2 L-type constraints
# EL(P): exp(1e-8*X1² + (X2+P)² + X3² + 4X4² + X5² + ... + X10²)
# F1: -U + EL(P=2)  ≤ 0
# F2: -U + EL(P=-2) ≤ 0
function polak2_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    start = T[T(100); fill(T(0.1), 10)]
    ExaModels.@add_var(c, x, 11; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_obj(c, x[11])
    ExaModels.@add_con(c,
        -x[11] + exp(T(1e-8)*x[1]^2 + (x[2] + T(2))^2
            + x[3]^2 + T(4)*x[4]^2 + x[5]^2 + x[6]^2
            + x[7]^2 + x[8]^2 + x[9]^2 + x[10]^2);
        lcon = T(-Inf), ucon = T(0))
    ExaModels.@add_con(c,
        -x[11] + exp(T(1e-8)*x[1]^2 + (x[2] - T(2))^2
            + x[3]^2 + T(4)*x[4]^2 + x[5]^2 + x[6]^2
            + x[7]^2 + x[8]^2 + x[9]^2 + x[10]^2);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
