# BDQRTIC: Banded diagonal quartic problem (scalable, unconstrained)
# N variables (default N=5000), all free, start = 1.0
# N-4 pairs of groups L(I) and G(I), for I=1..N-4:
#   L(I): linear group L2 type, linear = -4*X(I), constant RHS = -3
#     v = -4*X(I) - (-3) = -4*X(I) + 3, contribution = v^2
#   G(I): L2 group type with elements SQ(EVAR) = EVAR^2
#     v = 1*X(I)^2 + 2*X(I+1)^2 + 3*X(I+2)^2 + 4*X(I+3)^2 + 5*X(N)^2
#     contribution = v^2
# Source: Conn, Gould, Lescrenier & Toint (1988), problem 61
function bdqrtic_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    # L(I) groups: L2 type, v = -4*X(I) + 3, contribution = v^2
    ExaModels.@add_obj(c, (-T(4) * x[i] + T(3))^2 for i in 1:N-4)
    # G(I) groups: L2 type,
    # v = X(I)^2 + 2*X(I+1)^2 + 3*X(I+2)^2 + 4*X(I+3)^2 + 5*X(N)^2
    # contribution = v^2
    ExaModels.@add_obj(
        c,
        (x[i]^2 + T(2) * x[i+1]^2 + T(3) * x[i+2]^2 + T(4) * x[i+3]^2 +
         T(5) * x[N]^2)^2 for i in 1:N-4
    )
    return ExaModels.ExaModel(c; kwargs...)
end
