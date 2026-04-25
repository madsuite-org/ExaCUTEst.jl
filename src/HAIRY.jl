# HAIRY: Two-variable function with many local optima (Toint)
# 2 free variables: X1, X2; start = [-5, -7]
# Objective: 30*sin²(7*X1)*cos²(7*X2) + 100*sqrt(0.01+(X1-X2)²) + 100*sqrt(0.01+X1²)
# HLENGTH=30, CSLOPE=100, DENS=7, SMOOTH=0.01
# FUR: sin²(DENS*V1)*cos²(DENS*V2); DCUP: sqrt(SMOOTH+(V1-V2)²); 1CUP: sqrt(SMOOTH+V²)
function hairy_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(-5), T(-7)])
    ExaModels.@add_obj(c,
        T(30) * sin(T(7)*x[1])^2 * cos(T(7)*x[2])^2
        + T(100) * sqrt(T(0.01) + (x[1] - x[2])^2)
        + T(100) * sqrt(T(0.01) + x[1]^2))
    return ExaModels.ExaModel(c; kwargs...)
end
