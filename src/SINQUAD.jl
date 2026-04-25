# SINQUAD: Another function with nontrivial groups and repetitious elements (N=5000 variables)
# Source: N. Gould, Dec 1989
# SIF: N=5000 variables, all free (FR 'DEFAULT'), start=0.1
#   G1: linear X1*1.0, RHS=1.0, L4 group type → (X1-1)^4
#   G(I) for I=2..N: the second DO loop in GROUP USES (adding SINE element S(I) to G(2..N-1)
#     in a separate context) causes CUTEst's SIF decoder to reset the group type to identity
#     for those groups. So:
#     G(I) for I=2..N-1: identity → X(I)^2 - X(1)^2 + sin(X(I)-X(N))
#       (SINE element: IV U1 = V1-V2 = X(I)-X(N), F = sin(U1))
#     G(N): L2 type retained (not in second loop) → (X(N)^2 - X(1)^2)^2
# Empirically verified against CUTEst at multiple test points.
function sinquad_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.1))
    # G1: (X1 - 1)^4
    ExaModels.@add_obj(c, (x[1] - T(1))^4)
    # G(I) for I=2..N-1: identity group → x[i]^2 - x[1]^2 + sin(x[i] - x[N])
    ExaModels.@add_obj(c, x[i]^2 - x[1]^2 + sin(x[i] - x[N]) for i in 2:N-1)
    # G(N): L2 group → (x[N]^2 - x[1]^2)^2
    ExaModels.@add_obj(c, (x[N]^2 - x[1]^2)^2)
    return ExaModels.ExaModel(c; kwargs...)
end
