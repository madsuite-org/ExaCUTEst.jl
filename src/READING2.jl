# READING2: Linear optimal control (tide modelling), N=2000 default
# Source: Lyle & Nichols (1991)
# Variables: X1_(0..N), X2_(0..N), U(0..N) (3*(N+1) variables)
# SIF variable order: X1_(0), X2_(0), U(0), X1_(1), X2_(1), U(1), ...
# Julia indices: x[3i+1]=X1_(i), x[3i+2]=X2_(i), x[3i+3]=U(i)
#
# Objective (N type, linear):
#   sum_{I=1}^N [ -H/2*cos(2π*I*H)*X1_(I) + (-H/2*cos(2π*(I-1)*H))*X1_(I-1)
#                 + H/(8π²)*U(I) + H/(8π²)*U(I-1) ]
#
# Constraints C1_(I): X1_(I)/H - X1_(I-1)/H - 0.5*X2_(I) - 0.5*X2_(I-1) = 0
# Constraints C2_(I): X2_(I)/H - X2_(I-1)/H - 0.5*U(I) - 0.5*U(I-1) = 0
#
# Bounds: X1_(0)=0, X2_(0)=0 fixed; X1_(1..N) free; X2_(1..N) in [-0.125, 0.125]
#         U(0..N) in [-1, 1]
function reading2_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    PI  = T(3.141592653589)
    H   = T(1) / T(N)
    PI2 = PI^2
    H8PI2 = H / (T(8) * PI2)  # H/(8π²)

    c2pit = T[cos(T(2) * PI * T(i) * H) for i in 0:N]

    # Variable ordering: x[3i+1]=X1_(i), x[3i+2]=X2_(i), x[3i+3]=U(i)
    lvar = Vector{T}(undef, 3 * (N + 1))
    uvar = Vector{T}(undef, 3 * (N + 1))
    start = zeros(T, 3 * (N + 1))
    for i in 0:N
        j = 3i + 1
        # X1_(i)
        if i == 0
            lvar[j] = T(0); uvar[j] = T(0)
        else
            lvar[j] = T(-Inf); uvar[j] = T(Inf)
        end
        # X2_(i)
        if i == 0
            lvar[j + 1] = T(0); uvar[j + 1] = T(0)
        else
            lvar[j + 1] = T(-0.125); uvar[j + 1] = T(0.125)
        end
        # U(i)
        lvar[j + 2] = T(-1); uvar[j + 2] = T(1)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3 * (N + 1); lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, ct, c2pit)

    # Objective: sum_{I=1}^N [...]
    # X1_(I) = x[3I+1], X1_(I-1) = x[3(I-1)+1] = x[3I-2]
    # U(I) = x[3I+3], U(I-1) = x[3I]
    ExaModels.@add_obj(c,
        (-H / T(2)) * ct[d.I + 1] * x[3d.I + 1] +
        (-H / T(2)) * ct[d.I] * x[3d.I - 2] +
        H8PI2 * x[3d.I + 3] + H8PI2 * x[3d.I]
        for d in [(I = I,) for I in 1:N])

    # C1_(I): X1_(I)/H - X1_(I-1)/H - 0.5*X2_(I) - 0.5*X2_(I-1) = 0
    # X2_(I) = x[3I+2], X2_(I-1) = x[3I-1]
    ExaModels.@add_con(c,
        x[3d.I + 1] / H - x[3d.I - 2] / H -
        T(0.5) * x[3d.I + 2] - T(0.5) * x[3d.I - 1]
        for d in [(I = I,) for I in 1:N])

    # C2_(I): X2_(I)/H - X2_(I-1)/H - 0.5*U(I) - 0.5*U(I-1) = 0
    ExaModels.@add_con(c,
        x[3d.I + 2] / H - x[3d.I - 1] / H -
        T(0.5) * x[3d.I + 3] - T(0.5) * x[3d.I]
        for d in [(I = I,) for I in 1:N])

    return ExaModels.ExaModel(c; kwargs...)
end
