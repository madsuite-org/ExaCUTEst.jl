# ERRINRSM: Variable-dimension chained "error" Rosenbrock function, N=50 default
# Objective: sum_{I=2}^{N} [(x(I-1) - AI * x(I)^2)^2 + (x(I) - 1)^2]
# where AI = 16 * (sin(I) + 1.5)^2
# All variables free, start x(I) = -1
function errinrsm_model(N = 50; T = Float64, backend = nothing, kwargs...)
    x0 = fill(T(-1), N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # SQ(I) group: linear x(I-1) + element ETYPE(X(I)) = x(I-1) - x(I)^2
    # Element: ETYPE = -V1^2, so ELA(I) contributes -x(I)^2
    # Coefficient AI applied: group value = x(I-1) + AI*(-x(I)^2) = x(I-1) - AI*x(I)^2
    # L2: contribution = (x(I-1) - AI*x(I)^2)^2
    # B(I) group: linear x(I) - rhs=1
    # L2: contribution = (x(I) - 1)^2

    sq_data = [(im1 = i - 1, i = i, ai = T(16) * (sin(T(i)) + T(1.5))^2) for i in 2:N]

    ExaModels.@add_obj(c,
        (x[d.im1] - d.ai * x[d.i]^2)^2
        for d in sq_data)

    ExaModels.@add_obj(c,
        (x[i] - T(1))^2
        for i in 2:N)

    return ExaModels.ExaModel(c; kwargs...)
end
