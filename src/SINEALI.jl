# SINEALI: Sine-based variation on extended Rosenbrock (N=1000 variables, unconstrained)
# minimize sin(x[1]-1) + sum_{i=2}^{N} sin(x[i]-x[i-1]^2) / 0.01
# Source: Ali Bouriacha (private communication), SIF by Gould & Toint, 1993
# SIF: GROUP TYPE SIN (F=sin(v)), 'SCALE' 0.01 for groups SQ(2..N)
#   Contribution formula: F(v)/sigma = sin(v)/sigma
#   SQ(1): sin(x[1]-1) (no scale, constant RHS=1)
#   SQ(I): sin(x[I] - x[I-1]^2) / 0.01  (linear x[I], element XSQ(I-1)=-x[I-1]^2 coeff -1)
# Variable bounds via SIF parameter arithmetic (PI=3.1415926535):
#   X(1): uvar = PI/2, lvar = PI/2 - 2*PI
#   X(I): ub1 = prev_ub + PI/2, uvar = sqrt(ub1), lvar = uvar - 2*PI
# Default N = 1000 (matching SIF default)
function sineali_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    PI_val = 3.1415926535
    PI_half = PI_val * 0.5
    TWO_PI = PI_val * 2.0
    ub = PI_half
    lbvals = Float64[]
    ubvals = Float64[]
    push!(ubvals, ub)
    push!(lbvals, ub - TWO_PI)
    for i in 2:N
        ub1 = ub + PI_half
        ub = sqrt(ub1)
        push!(ubvals, ub)
        push!(lbvals, ub - TWO_PI)
    end
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T.(lbvals), uvar = T.(ubvals), start = T(0.0))
    ExaModels.@add_obj(c, sin(x[1] - T(1)))
    ExaModels.@add_obj(c, sin(x[i] - x[i-1]^2) / T(0.01) for i in 2:N)
    return ExaModels.ExaModel(c; kwargs...)
end
