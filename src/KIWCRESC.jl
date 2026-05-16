# KIWCRESC: Nonlinear minmax problem in two variables (Kiwiel 1985)
# min U  s.t.
#   U - x2 - (x1-0)^2 - (x2-1)^2 >= 0  → F1: U - x2 + (x1)^2 + (x2-1)^2 >= 0... wait
# Let me re-read the SIF:
#   OBJ: U (minimize)
#   F1 (L type, lcon>=0): U - 1.0 with element X1SQ(V=X1,S=0) coeff 1.0 and X2SQ(V=X2,S=1) coeff 1.0
#     SSQ element: (V-S)^2 → X1SQ = (X1-0)^2 = X1^2, X2SQ = (X2-1)^2
#     F1 linear part: U - 1.0, constant = 1.0 (on RHS), so group value = U - x2 + X1SQ + X2SQ - 1.0
#     Wait: F1 group: U coeff -1.0, x2 coeff 1.0, constant 1.0
#     Elements X1SQ coeff 1.0, X2SQ coeff 1.0
#     Group value = -U + x2 + X1^2 + (X2-1)^2 - 1.0
#     L type: v <= 0 → -U + x2 + X1^2 + (X2-1)^2 - 1.0 <= 0
#     → U - x2 - X1^2 - (X2-1)^2 + 1.0 >= 0
#   F2 (L type): U - (-1.0) with element X1SQ coeff -1.0, X2SQ coeff -1.0
#     Group value = -U + x2 - X1^2 - (X2-1)^2 - (-1.0) = -U + x2 - X1^2 - (X2-1)^2 + 1.0
#     L type: v <= 0 → -U + x2 - X1^2 - (X2-1)^2 + 1.0 <= 0
#     → U - x2 + X1^2 + (X2-1)^2 - 1.0 >= 0
# So:
#   F1: U >= x2 + X1^2 + (X2-1)^2 - 1.0
#   F2: U >= x2 - X1^2 - (X2-1)^2 + 1.0
# All variables free. Start: X1=-1.5, X2=2.0, U not specified → 0
# Variable order: X1 (1st), X2 (2nd), U (3rd)
function kiwcresc_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # X1, X2, U - all free
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf),
                       start = T[-1.5, 2.0, 0.0])

    # Minimize U = x[3]
    ExaModels.@add_obj(c, x[3])

    # F1 (L): group value <= 0:  -x[3] + x[2] + x[1]^2 + (x[2]-1)^2 - 1 <= 0
    # Residual form: x[3] - x[2] - x[1]^2 - (x[2]-1)^2 + 1 >= 0
    ExaModels.@add_con(c, x[3] - x[2] - x[1]^2 - (x[2] - T(1))^2 + T(1);
                       lcon = T(0), ucon = T(Inf))

    # F2 (L): group value <= 0:  -x[3] + x[2] - x[1]^2 - (x[2]-1)^2 + 1 <= 0
    # Residual form: x[3] - x[2] + x[1]^2 + (x[2]-1)^2 - 1 >= 0
    ExaModels.@add_con(c, x[3] - x[2] + x[1]^2 + (x[2] - T(1))^2 - T(1);
                       lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
