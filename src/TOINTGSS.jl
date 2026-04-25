# TOINTGSS: Toint's Gaussian problem (N variables, unconstrained)
# minimize sum_{i=1}^{N-2} (AP + x[i+2]^2) * (2 - exp(-（x[i]-x[i+1])^2 / (0.1 + x[i+2]^2)))
# All variables free, starting point x = 3.0
# Default N = 5000 (matching SIF default IE N = 5000)
# AP = 10.0 / (N - 2)
#
# SIF structure:
#   N-2 groups G(I), each using element E(I) of type TG (no group type = identity)
#   TG element: IV U1 = V1*1.0 + V2*(-1.0), IV U2 = V3; V1=X(I), V2=X(I+1), V3=X(I+2)
#   F = (AP + U2^2) * (2 - exp(-U1^2 / (0.1 + U2^2)))
function tointgss_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    AP = T(10.0) / T(N - 2)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(3.0))
    ExaModels.@add_obj(
        c,
        let u1 = x[i] - x[i+1], u2 = x[i+2], t = T(0.1) + u2^2
            (AP + u2^2) * (T(2) - exp(-u1^2 / t))
        end
        for i in 1:N-2
    )
    return ExaModels.ExaModel(c; kwargs...)
end
