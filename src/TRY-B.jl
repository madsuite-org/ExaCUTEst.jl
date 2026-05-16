# TRY-B: Test problem with variable absent from objective (QQR2-AN-2-1)
# minimize (x - 1)^2  s.t. (x - 1)^2 + (y - 10)^2 = 1
# SIF: OBJ group has linear X*1.0, constant 1.0, GROUP TYPE SQUARE → (X-1)^2
#      CON group: elements (X-1)^2 + (Y-10)^2, constant 1.0
# Variables default to x >= 0 (no BOUNDS section)
function try_b_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[0, 0], uvar = T[Inf, Inf], start = T[10, 10])
    ExaModels.@add_obj(c, (x[1] - T(1))^2)
    ExaModels.@add_con(c, (x[1] - T(1))^2 + (x[2] - T(10))^2 - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
