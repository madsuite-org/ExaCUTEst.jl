# BDQRTICNE: Banded diagonal quartic, nonlinear equations version
# N variables, 2*(N-4) equality constraints
# L(I): -4*x[i] + 3 = 0                                        for i=1..N-4
# G(I): x[i]^2 + 2*x[i+1]^2 + 3*x[i+2]^2 + 4*x[i+3]^2 + 5*x[N]^2 = 0  for i=1..N-4
# All variables free; start: x[i] = 1.0
function bdqrticne_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; start = T(1.0))
    # L(I) groups: -4*x[i] + 3 = 0
    ExaModels.@add_con(c, -T(4) * x[i] + T(3) for i in 1:N-4)
    # G(I) groups: x[i]^2 + 2*x[i+1]^2 + 3*x[i+2]^2 + 4*x[i+3]^2 + 5*x[N]^2 = 0
    ExaModels.@add_con(
        c,
        x[i]^2 + T(2) * x[i+1]^2 + T(3) * x[i+2]^2 + T(4) * x[i+3]^2 +
        T(5) * x[N]^2 for i in 1:N-4
    )
    return ExaModels.ExaModel(c; kwargs...)
end
