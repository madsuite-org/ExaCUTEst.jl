# ARTIF: Artificial nonlinear system (Irani et al. 1991)
# N=5000 free variables X(1)..X(N), plus X(0)=X(N+1)=0 (fixed boundaries)
# In ExaModels: x[1]=X(0), x[i+1]=X(i) for i=1..N, x[N+2]=X(N+1)
#
# Element TRIG(X, PAR): F = atan(sin(mod(PAR,100) * X))
# Constraints EQ(I): atan(sin(mod(I,100)*X(I))) - 0.05*X(I-1) - 0.05*X(I) - 0.05*X(I+1) = 0
#   In Julia: atan(sin(FACT[i]*x[i+1])) - 0.05*x[i] - 0.05*x[i+1] - 0.05*x[i+2] = 0
#
# @add_par for FACT ensures all N constraints share one compiled type.
function artif_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    fact = T[T(mod(i, 100)) for i in 1:N]

    lvar = fill(T(-Inf), N + 2)
    uvar = fill(T(Inf), N + 2)
    lvar[1]     = uvar[1]     = T(0)  # X(0) = 0 fixed
    lvar[N + 2] = uvar[N + 2] = T(0)  # X(N+1) = 0 fixed

    start = fill(T(1), N + 2)
    start[1]     = T(0)
    start[N + 2] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N + 2; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, FACT, fact)
    ExaModels.@add_con(c,
        atan(sin(FACT[i] * x[i + 1])) - T(0.05) * x[i] - T(0.05) * x[i + 1] -
        T(0.05) * x[i + 2] for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
