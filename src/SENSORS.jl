# SENSORS: 2D optimal sensor placement, N=100 default
# Source: Zhang & Wang (1993); SIF input by N. Gould, 1994
# N variables THETA(1..N), all free
# Objective: sum_{I,J=1}^{N} -(sin(θI)*sin(θJ)*sin(θI-θJ))^2
#   (N type groups with -L2 group type applied to SINFUN elements)
# Unconstrained (classification SUR2)
function sensors_model(N = 100; T = Float64, backend = nothing, kwargs...)
    start = T[T(i) / T(N) for i in 1:N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N;
                       lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: sum_{I=1..N} sum_{J=1..N} -(sin(θI)*sin(θJ)*sin(θI-θJ))^2
    data = [(I = I, J = J) for I in 1:N for J in 1:N]
    ExaModels.@add_obj(c,
        -(sin(x[d.I]) * sin(x[d.J]) * sin(x[d.I] - x[d.J]))^2
        for d in data)

    return ExaModels.ExaModel(c; kwargs...)
end
