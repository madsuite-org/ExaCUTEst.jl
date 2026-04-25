# NONCVXU2: Nonconvex unconstrained function (scalable, N=5000)
# minimize sum_{i=1}^{N} [ (x[i] + x[j2] + x[j3])^2 + 4*cos(x[i] + x[j2] + x[j3]) ]
# where j2 = mod(3i-2, N) + 1, j3 = mod(7i-3, N) + 1  (1-based indices)
# All variables free. Start: X(I) = I (real value).
# Source: Nick Gould, April 1996
# SIF: Single OBJ group; element SQ: IV V = V1+V2+V3, F = V^2;
#      element COS: IV V = V1+V2+V3, F = 4*cos(V)
#   V1=X(I), V2=X(j2), V3=X(j3)
function noncvxu2_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                      start = T[T(i) for i in 1:N])
    # Precompute index pairs
    j2 = [mod(3*i - 2, N) + 1 for i in 1:N]
    j3 = [mod(7*i - 3, N) + 1 for i in 1:N]
    ExaModels.@add_par(c, pj2, j2)
    ExaModels.@add_par(c, pj3, j3)
    ExaModels.@add_obj(c, (x[i] + x[Int(pj2[i])] + x[Int(pj3[i])])^2 +
                          T(4) * cos(x[i] + x[Int(pj2[i])] + x[Int(pj3[i])])
                       for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
