# SPIRAL: Nonlinear minmax (Polak, Higgins & Mayne 1992)
# 3 free variables: X1, X2, U; start = [1.41831, -4.79462, 1.0]
# Objective: U; 2 G-type constraints
# R = sqrt(X1²+X2²)
# C1: U - (X1 - R*cos(R))² - 0.005*X1² - 0.005*X2² ≥ 0
# C2: U - (X2 - R*sin(R))² - 0.005*X1² - 0.005*X2² ≥ 0
function spiral_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(1.41831), T(-4.79462), T(1.0)])
    ExaModels.@add_obj(c, x[3])
    ExaModels.@add_con(c,
        x[3] - (x[1] - sqrt(x[1]^2 + x[2]^2)*cos(sqrt(x[1]^2 + x[2]^2)))^2
        - T(0.005)*x[1]^2 - T(0.005)*x[2]^2;
        lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c,
        x[3] - (x[2] - sqrt(x[1]^2 + x[2]^2)*sin(sqrt(x[1]^2 + x[2]^2)))^2
        - T(0.005)*x[1]^2 - T(0.005)*x[2]^2;
        lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
