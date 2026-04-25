# POLAK3: Nonlinear minmax in eleven variables (Polak, Mayne & Higgins 1991)
# 12 free variables: X(1..11), U; start = 1.0
# Objective: U; 10 L-type constraints F(I), I=1..10
# F(I): -U + sum_{J=1..11} (1/J) * exp((X(J) - sin((I-1)+2*J))²) ≤ 0
# Note: SIF has RI RJ-1 J (not J-1), so B = float(J) → sin(A + 2*B) = sin((I-1)+2*J)
function polak3_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 12; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c, x[12])
    ExaModels.@add_con(c,
        -x[12] + sum(exp((x[j] - sin(T(i-1) + T(2)*T(j)))^2) / T(j) for j in 1:11)
        for i in 1:10; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
