# INDEFM: modified INDEF variant by Luksan et al (N=100000 variables, unconstrained)
# minimize 100*sum_{i=1}^{N} sin(0.01*x[i]) + 0.5*sum_{i=2}^{N-1} cos(2*x[i]-x[N]-x[1])
# Source: Luksan, Matonoha, Vlcek (2010), problem 37; based on Gould (1992)
# SIF: SIN(I) groups with SIN type (100*sin(0.01*v)) for i=1..N;
#   COS(I) groups with COS type (ALPHA*cos(v)) for i=2..N-1, ALPHA=0.5
# All variables FREE; start: x[i] = i/(N+1)
function indefm_model(N = 100000; T = Float64, backend = nothing, kwargs...)
    start_vec = T[i / (N + 1) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start_vec)
    ExaModels.@add_obj(c, T(100) * sin(T(0.01) * x[i]) for i in 1:N)
    ExaModels.@add_obj(c, T(0.5) * cos(T(2) * x[i] - x[N] - x[1]) for i in 2:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
