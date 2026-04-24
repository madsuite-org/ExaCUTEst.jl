# HS5: Hock-Schittkowski problem 5 (2 variables, bound constrained)
# minimize sin(x1+x2) + (x1-x2)^2 - 1.5*x1 + 2.5*x2 + 1
# bounds: -1.5 <= x1 <= 4, -3 <= x2 <= 3
# Source: Hock & Schittkowski (1981), problem 5
# SIF: G1=SINE(x1+x2), G2=SQ(x1-x2), G3=identity(-1.5*x1+2.5*x2+1) (G3 constant=-1 → value+1)
function hs5_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-1.5, -3.0], uvar = T[4.0, 3.0])
    ExaModels.@add_obj(c, sin(x[1] + x[2]) + (x[1] - x[2])^2 + (-1.5 * x[1] + 2.5 * x[2] + 1))
    return ExaModels.ExaModel(c; kwargs...)
end
