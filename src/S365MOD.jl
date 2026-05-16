# S365MOD: Modified Walukiewicz problem (like S365 but NEARZERO=0.5 affects bounds)
# Source: Schittkowski "More Test Examples for NLP Codes", modified problem 365
# Identical structure to S365 but:
#   x1 >= NEARZERO = 0.5 (instead of >= 0)
#   x3 >= NEARZERO = 0.5 (instead of >= 0)
function s365mod_model(; T = Float64, backend = nothing, kwargs...)
    NEARZERO = T(0.5)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7;
                       lvar = T[NEARZERO, -Inf, NEARZERO, -Inf, 1.0, -Inf, 1.0],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[3.0, 0.0, 2.0, -1.5, 1.5, 5.0, 0.0])
    # Objective: x1*x3
    ExaModels.@add_obj(c, x[1] * x[3])
    # C1: (x4-x6)^2 + (x5-x7)^2 - 4 >= 0
    ExaModels.@add_con(c, (x[4] - x[6])^2 + (x[5] - x[7])^2 - T(4);
                       lcon = T(0), ucon = T(Inf))
    # C2: x3*x4 - x2*x5 - |x2| - x3^2 >= 0
    ExaModels.@add_con(c, x[3] * x[4] - x[2] * x[5] - sqrt(x[2]^2) - x[3]^2;
                       lcon = T(0), ucon = T(Inf))
    # C3: x3*x6 - x2*x7 - |x2| - x3^2 >= 0
    ExaModels.@add_con(c, x[3] * x[6] - x[2] * x[7] - sqrt(x[2]^2) - x[3]^2;
                       lcon = T(0), ucon = T(Inf))
    # C4: x1*x3 + x2*x5 - x1*x5 - x3*x4 - |x3| - (x2-x1)^2 >= 0
    ExaModels.@add_con(c,
        x[1] * x[3] + x[2] * x[5] - x[1] * x[5] - x[3] * x[4] -
        sqrt(x[3]^2) - (x[2] - x[1])^2;
        lcon = T(0), ucon = T(Inf))
    # C5: x1*x3 + x2*x7 - x1*x7 - x3*x6 - |x3| - (x2-x1)^2 >= 0
    ExaModels.@add_con(c,
        x[1] * x[3] + x[2] * x[7] - x[1] * x[7] - x[3] * x[6] -
        sqrt(x[3]^2) - (x[2] - x[1])^2;
        lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
