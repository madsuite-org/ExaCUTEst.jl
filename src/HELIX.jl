# HELIX: Helical valley problem (3 variables, unconstrained)
# Source: More', Garbow, Hillstrom (1981), problem 7
# minimize f = (x3 - 10*TWOPII*atan(x2,x1))^2 / 0.01
#            + (sqrt(x1^2+x2^2) - 1)^2 / 0.01
#            + x3^2
# where TWOPII = 1/(2*pi) ≈ 0.15915494
# Groups A, B, C all use L2 group type (contribution = v^2 / sigma)
# Group A: sigma=0.01, v = x3 + (-10)*TWOPII*atan(x2,x1)
# Group B: sigma=0.01, v = sqrt(x1^2+x2^2) - 1.0  (CONSTANT=1.0)
# Group C: sigma=1 (no scale), v = x3
function helix_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # All variables free (FR DEFAULT), start: X1=-1, X2=0, X3=0
    ExaModels.@add_var(c, x, 3; lvar = T[-Inf, -Inf, -Inf], uvar = T[Inf, Inf, Inf],
                       start = T[-1.0, 0.0, 0.0])
    # TWOPII = 0.15915494 in SIF, stored as Fortran float32 → 0.15915493667125702
    TWOPII = T(Float32(0.15915494))
    # Group A (sigma=0.01): v = x3 - 10 * TWOPII * atan(x2, x1)
    ExaModels.@add_obj(c, (x[3] - T(10) * TWOPII * atan(x[2], x[1]))^2 / T(0.01))
    # Group B (sigma=0.01): v = sqrt(x1^2 + x2^2) - 1.0
    ExaModels.@add_obj(c, (sqrt(x[1]^2 + x[2]^2) - T(1))^2 / T(0.01))
    # Group C (sigma=1): v = x3
    ExaModels.@add_obj(c, x[3]^2)
    return ExaModels.ExaModel(c; kwargs...)
end
