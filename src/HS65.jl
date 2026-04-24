function hs65_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # Start declared as [-5, 5, 0] but clamped to bounds [-4.5, 4.5]
    ExaModels.@add_var(c, x, 3;
                       lvar = T[-4.5, -4.5, -5.0],
                       uvar = T[4.5, 4.5, 5.0],
                       start = T[-4.5, 4.5, 0.0])
    # OBJ1 (L2, no scale): (x1-x2)²
    # OBJ2 (L2, scale=9, constant=10): (x1+x2-10)²/9
    # OBJ3 (L2, no scale, constant=5): (x3-5)²
    ExaModels.@add_obj(c,
        (x[1]-x[2])^2 + (x[1]+x[2]-10)^2/9 + (x[3]-5)^2)
    # C1 (G): 48 - x1²-x2²-x3² ≥ 0
    ExaModels.@add_con(c, 48 - x[1]^2 - x[2]^2 - x[3]^2; lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
