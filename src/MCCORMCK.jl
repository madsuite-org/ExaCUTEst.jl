# MCCORMCK: Extended McCormick bounded problem (Toint 1983)
# N=5000 variables in [-1.5, 3.0]; start = 0
# Objective = sum_{I=1..N-1} [(X(I+1)-X(I))^2 + sin(X(I)+X(I+1)) - 1.5*X(I) + 2.5*X(I+1) + 1]
# Group E(I): linear = -1.5*X(I)+2.5*X(I+1), rhs=-1; elements SQUARE(X(I+1)-X(I)) + SINE(X(I)+X(I+1))
function mccormck_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-1.5), uvar = T(3.0), start = T(0))
    ExaModels.@add_obj(c,
        (x[i+1] - x[i])^2 + sin(x[i] + x[i+1]) - T(1.5)*x[i] + T(2.5)*x[i+1] + T(1)
        for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
