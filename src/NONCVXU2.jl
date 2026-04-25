# NONCVXU2: Nonconvex unconstrained function (scalable, N=5000)
# minimize sum_{i=1}^{N} [ (x[i] + x[j2(i)] + x[j3(i)])^2 + 4*cos(x[i] + x[j2(i)] + x[j3(i)]) ]
# where j2(i) = mod(3i-2, N) + 1, j3(i) = mod(7i-3, N) + 1  (1-based)
# All variables free. Start: X(I) = I (real value).
# Source: Nick Gould, April 1996
# SIF: Single OBJ group; element SQ (IV V = V1+V2+V3, F = V^2)
#      and element COS (IV V = V1+V2+V3, F = 4*cos(V))
#   V1=X(I), V2=X(j2(I)), V3=X(j3(I))
function noncvxu2_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    # Precompute index triples as NamedTuple array (integer fields used for variable indexing)
    data = [(i = i, j2 = mod(3 * i - 2, N) + 1, j3 = mod(7 * i - 3, N) + 1) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                      start = T[T(i) for i in 1:N])
    # SQ element: (x[i] + x[j2] + x[j3])^2
    ExaModels.@add_obj(c, (x[d.i] + x[d.j2] + x[d.j3])^2 for d in data)
    # COS element: 4*cos(x[i] + x[j2] + x[j3])
    ExaModels.@add_obj(c, T(4) * cos(x[d.i] + x[d.j2] + x[d.j3]) for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
