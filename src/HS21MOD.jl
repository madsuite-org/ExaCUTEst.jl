# HS21MOD: HS21 modification with 7 variables (Hock & Schittkowski 1981)
# X1∈[2,50], X2∈[-50,50], X3∈(-∞,50], X4∈[2,∞), X5 free, X6∈(-∞,0], X7∈[0,∞)
# start = [-1,-1,0,0,0,0,0]; Objective: 0.01X1²+X2²+0.01X3²+X4²+0.01X5²+0.01X6²+X7²-100
# Constraint: 10X1 - X2 - 10 ≥ 0
function hs21mod_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7;
        lvar = T[T(2), T(-50), T(-Inf), T(2), T(-Inf), T(-Inf), T(0)],
        uvar = T[T(50), T(50), T(50), T(Inf), T(Inf), T(0), T(Inf)],
        start = T[T(-1), T(-1), T(0), T(0), T(0), T(0), T(0)])
    ExaModels.@add_obj(c,
        T(0.01)*x[1]^2 + x[2]^2 + T(0.01)*x[3]^2 + x[4]^2
        + T(0.01)*x[5]^2 + T(0.01)*x[6]^2 + x[7]^2 - T(100))
    ExaModels.@add_con(c, T(10)*x[1] - x[2] - T(10); lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
