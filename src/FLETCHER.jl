# FLETCHER: Least area of right-angled triangle containing unit circle, 4 vars, 4 constraints
# Variables: X1,X2,X3,X4 (all free except X4>=1); Start: all=1
# Objective: X1*X2
# C1 (E): (X1*X3+X2*X4)^2/(X1^2+X2^2) - X3^2 - X4^2 + 1 = 0
# C2 (G): X1 - X3 - 1 >= 0
# C3 (G): X2 - X4 - 1 >= 0
# C4 (G): X3 - X4 >= 0
function fletcher_model(; T = Float64, backend = nothing, kwargs...)
    NV = 4
    x0 = ones(T, NV)
    lvar = T[-Inf, -Inf, -Inf, 1]
    uvar = fill(T(Inf), NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: X1*X2
    ExaModels.@add_obj(c, x[1] * x[2])

    # C1 (equality): (X1*X3+X2*X4)^2/(X1^2+X2^2) - X3^2 - X4^2 + 1 = 0
    ExaModels.@add_con(c,
        (x[1] * x[3] + x[2] * x[4])^2 / (x[1]^2 + x[2]^2) - x[3]^2 - x[4]^2 + T(1))

    # C2 (G>=0): X1 - X3 - 1 >= 0 → lcon=0, ucon=Inf
    ExaModels.@add_con(c, x[1] - x[3] - T(1); lcon = T(0), ucon = T(Inf))

    # C3 (G>=0): X2 - X4 - 1 >= 0
    ExaModels.@add_con(c, x[2] - x[4] - T(1); lcon = T(0), ucon = T(Inf))

    # C4 (G>=0): X3 - X4 >= 0
    ExaModels.@add_con(c, x[3] - x[4]; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
