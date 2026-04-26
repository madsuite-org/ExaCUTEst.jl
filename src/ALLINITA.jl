# ALLINITA: "All in it" constrained problem - ALLINIT with 4 constraints added (Gould 2013)
# 4 variables (same bounds/start as ALLINIT), 4 constraints (2 linear, 2 nonlinear)
# Objective: identical to ALLINIT (FT groups identity, FNT groups L2)
# CUTEst ordering: linear L1, L2 before nonlinear C1, C2
function allinita_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar = T[T(-Inf), T(1), T(-1e10), T(2)],
        uvar = T[T(Inf), T(Inf), T(1), T(2)],
        start = T[T(0), T(0), T(0), T(0)])
    # Objective (same as ALLINIT)
    ExaModels.@add_obj(c, x[3] - T(1))
    ExaModels.@add_obj(c, x[1]^2)
    ExaModels.@add_obj(c, x[2]^2 + (x[3]+x[4])^2)
    ExaModels.@add_obj(c, x[4] + sin(x[3])^2 + x[1]^2*x[2]^2 - T(3))
    ExaModels.@add_obj(c, sin(x[3])^2)
    ExaModels.@add_obj(c, (x[4] - T(1))^2)
    ExaModels.@add_obj(c, x[2]^4)
    ExaModels.@add_obj(c, (x[3]^2 + (x[4]+x[1])^2)^2)
    ExaModels.@add_obj(c, (x[1] + sin(x[4])^2 + x[2]^2*x[3]^2 - T(4))^2)
    ExaModels.@add_obj(c, sin(x[4])^4)
    # CUTEst constraint ordering: linear groups prepended (LIFO), nonlinear appended.
    # SIF defines: C1 (E,NL), C2 (G,NL), L1 (L,linear), L2 (E,linear).
    # Processing: C1→append, C2→append, L1→prepend→[L1,C1,C2], L2→prepend→[L2,L1,C1,C2].
    # L2 (E, rhs=0.25): X1+X3 = 0.25
    ExaModels.@add_con(c, x[1] + x[3] - T(0.25))
    # L1 (L, rhs=1.5): X1+X2+X3 ≤ 1.5
    ExaModels.@add_con(c, x[1] + x[2] + x[3] - T(1.5); lcon = T(-Inf), ucon = T(0))
    # C1 (E, rhs=1.0): X1² + X2² = 1
    ExaModels.@add_con(c, x[1]^2 + x[2]^2 - T(1))
    # C2 (G, rhs=0): X2² + X3² ≥ 0
    ExaModels.@add_con(c, x[2]^2 + x[3]^2; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
