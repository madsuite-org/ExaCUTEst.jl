# ERRINROS: Chained Rosenbrock variant (Toint 1990)
# N=50 free variables, no constraints
# Objective = sum_{I=2..N} [(X(I-1) - 16*ALPH(I)^2 * X(I)^2)^2 + (X(I)-1)^2]
# Start: all -1
function errinros_model(N = 50; T = Float64, backend = nothing, kwargs...)
    alph = T[1.25, 1.40, 2.40, 1.40, 1.75, 1.20, 2.25, 1.20, 1.00, 1.10,
             1.50, 1.60, 1.25, 1.25, 1.20, 1.20, 1.40, 0.50, 0.50, 1.25,
             1.80, 0.75, 1.25, 1.40, 1.60, 2.00, 1.00, 1.60, 1.25, 2.75,
             1.25, 1.25, 1.25, 3.00, 1.50, 2.00, 1.25, 1.40, 1.80, 1.50,
             2.20, 1.40, 1.50, 1.25, 2.00, 1.50, 1.25, 1.40, 0.60, 1.50]
    data_I = [(i = i, ai = T(16) * alph[i]^2) for i in 2:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(-1))
    ExaModels.@add_obj(c, (x[d.i-1] - d.ai * x[d.i]^2)^2 for d in data_I)
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
