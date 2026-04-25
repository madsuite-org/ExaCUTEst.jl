# ALLINITU: "All in it" verification problem (4 free variables, unconstrained)
# Source: N. Gould, private communication. SIF input: Nick Gould, June 1990.
#
# Group types: TRIVIAL (F=v), L2 (F=v^2)
# Element types: SQR(X)=X^2, SQR2(Y,Z) with IV X=Y+Z → X^2=(Y+Z)^2,
#                SINSQR(X)=sin(X)^2, PRODSQR(X,Y)=X^2*Y^2
#
# TRIVIAL groups (FT*) — each contributes its raw argument v:
#   FT1: v=0 → 0 (skip)
#   FT2: v = X3 - 1
#   FT3: v = X1^2
#   FT4: v = X2^2 + (X3+X4)^2
#   FT5: v = sin(X3)^2 + X1^2*X2^2 + X4 - 3
#   FT6: v = sin(X3)^2
#
# L2 groups (FNT*) — each contributes v^2:
#   FNT1: v=0 → 0 (skip)
#   FNT2: v = X4-1 → (X4-1)^2
#   FNT3: v = X2^2 → X2^4
#   FNT4: v = X3^2 + (X4+X1)^2 → (X3^2+(X4+X1)^2)^2
#   FNT5: v = sin(X4)^2 + X2^2*X3^2 + X1 - 4 → (...)^2
#   FNT6: v = sin(X4)^2 → sin(X4)^4
function allinitu_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T[-Inf, -Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf, Inf], start = T[0.0, 0.0, 0.0, 0.0])

    # TRIVIAL groups
    ExaModels.@add_obj(c, x[3] - T(1))                                         # FT2
    ExaModels.@add_obj(c, x[1]^2)                                               # FT3
    ExaModels.@add_obj(c, x[2]^2 + (x[3] + x[4])^2)                           # FT4
    ExaModels.@add_obj(c, sin(x[3])^2 + x[1]^2 * x[2]^2 + x[4] - T(3))       # FT5
    ExaModels.@add_obj(c, sin(x[3])^2)                                          # FT6

    # L2 groups
    ExaModels.@add_obj(c, (x[4] - T(1))^2)                                      # FNT2
    ExaModels.@add_obj(c, x[2]^4)                                                # FNT3: (X2^2)^2
    ExaModels.@add_obj(c, (x[3]^2 + (x[4] + x[1])^2)^2)                        # FNT4
    ExaModels.@add_obj(c, (sin(x[4])^2 + x[2]^2 * x[3]^2 + x[1] - T(4))^2)    # FNT5
    ExaModels.@add_obj(c, sin(x[4])^4)                                           # FNT6: (sin(X4)^2)^2

    return ExaModels.ExaModel(c; kwargs...)
end
