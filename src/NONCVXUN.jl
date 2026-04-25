# NONCVXUN: Nonconvex unconstrained function with unique minimum (N=5000)
# Source: Nick Gould, April 1996
# SIF classification: OUR2-AN-V-0
#
# Element SQ(I):  IV V = V1+V2+V3;  F = V^2 = (V1+V2+V3)^2
# Element COS(I): IV V = V1+V2+V3;  F = 4*cos(V) = 4*cos(V1+V2+V3)
# V1 = X(I), V2 = X(j1(I)), V3 = X(j2(I))
# j1(I) = mod(2*I - 1, N) + 1
# j2(I) = mod(3*I - 1, N) + 1
#
# Objective = sum_{I=1}^{N} [ (x[I] + x[j1(I)] + x[j2(I)])^2
#                            + 4*cos(x[I] + x[j1(I)] + x[j2(I)]) ]
# Bounds: all free (XR DEFAULT)
# Start: X(I) = float(I)
function noncvxun_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    data = [(i = i, j1 = mod(2 * i - 1, N) + 1, j2 = mod(3 * i - 1, N) + 1) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf),
                      start = T[T(i) for i in 1:N])
    # SQ element: (x[i] + x[j1] + x[j2])^2
    ExaModels.@add_obj(c, (x[d.i] + x[d.j1] + x[d.j2])^2 for d in data)
    # COS element: 4*cos(x[i] + x[j1] + x[j2])
    ExaModels.@add_obj(c, T(4) * cos(x[d.i] + x[d.j1] + x[d.j2]) for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
