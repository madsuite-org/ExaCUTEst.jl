# SINEVAL: Trigonometric Rosenbrock variant (2 variables, all free)
# G1: L2, scale=C=0.0001, linear X2 coeff 1.0, element SINE(X1) coeff -1.0
#   v = X2 - sin(X1), contribution = (X2-sin(X1))^2 / C = 10000*(X2-sin(X1))^2
# G2: L2, scale=4.0, linear X1 coeff 1.0, no RHS
#   v = X1, contribution = X1^2 / 4
# Start: X1=3*pi/2 ≈ 4.712389, X2=-1.0
function sineval_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[T(4.712389), T(-1.0)])
    ExaModels.@add_obj(c, (x[2] - sin(x[1]))^2 / T(0.0001))
    ExaModels.@add_obj(c, x[1]^2 / T(4))
    return ExaModels.ExaModel(c; kwargs...)
end
