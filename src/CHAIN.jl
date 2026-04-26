# CHAIN: Minimal potential energy of a hanging chain
# NH=400 intervals → 2*(NH+1) variables, NH+1 constraints (NH D-constraints + 1 LENGTH)
#
# SIF declares variables interleaved: X(0),U(0),X(1),U(1),...,X(NH),U(NH)
# Bounds: XR DEFAULT (lvar=0) + ZX X(0)=A, ZX X(NH)=B → fixed vars reordered to end
#
# CUTEst ordering:
#   U(0)        → pos 1
#   X(I)        → pos 2*I       for I=1..NH-1
#   U(I)        → pos 2*I+1     for I=1..NH-1
#   U(NH)       → pos 2*NH
#   X(0)        → pos nv-1   (first ZX)
#   X(NH)       → pos nv     (second ZX)
#
# Constraints in SIF order: D(0..NH-1) [linear], LENGTH [nonlinear]
function chain_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    A    = T(1)
    B    = T(3)
    L    = T(4)
    TMIN = T(0.25)
    h    = T(1) / T(NH)
    h2   = h / T(2)
    c4absBA = T(4) * abs(B - A)   # = 8.0

    nv = 2 * (NH + 1)

    xidx(k) = k == 0 ? nv - 1 : k == NH ? nv : 2 * k
    uidx(k) = k == 0 ? 1 : k == NH ? 2 * NH : 2 * k + 1

    lvar = fill(T(0), nv)   # XR DEFAULT: lvar=0
    uvar = fill(T(Inf), nv)
    lvar[nv-1] = A; uvar[nv-1] = A   # X(0) fixed
    lvar[nv]   = B; uvar[nv]   = B   # X(NH) fixed

    start = zeros(T, nv)
    for k in 0:NH
        t = T(k) / T(NH)
        start[xidx(k)] = c4absBA * t * (T(0.5) * t - TMIN) + A
        start[uidx(k)] = c4absBA * (t - TMIN)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: trapezoidal PE = H*X*sqrt(1+U^2)
    ExaModels.@add_obj(c, h2 * x[nv-1] * sqrt(T(1) + x[1]^2))
    ExaModels.@add_obj(c, h * x[2*i] * sqrt(T(1) + x[2*i+1]^2) for i in 1:NH-1)
    ExaModels.@add_obj(c, h2 * x[nv] * sqrt(T(1) + x[2*NH]^2))

    # D(I) constraints (purely linear): X(I)-X(I+1)+H/2*(U(I)+U(I+1))=0 for I=0..NH-1
    ExaModels.@add_con(c, x[nv-1] - x[2] + h2 * (x[1] + x[3]))           # D(0)
    ExaModels.@add_con(c, x[2*i] - x[2*i+2] + h2 * (x[2*i+1] + x[2*i+3]) for i in 1:NH-2)
    ExaModels.@add_con(c, x[2*(NH-1)] - x[nv] + h2 * (x[2*(NH-1)+1] + x[2*NH]))  # D(NH-1)

    # LENGTH constraint (nonlinear): trapezoidal sum sqrt(1+U^2) - L = 0
    ExaModels.@add_con(c,
        h2 * sqrt(T(1) + x[1]^2) +
        sum(h * sqrt(T(1) + x[2*i+1]^2) for i in 1:NH-1) +
        h2 * sqrt(T(1) + x[2*NH]^2) - L
    )

    return ExaModels.ExaModel(c; kwargs...)
end
