# QPBAND: Banded QP
# N vars, bounds 0 <= x <= 2
# Objective (N type): sum_i (-i/N)*x_i + QUADRATIC terms
# QUADRATIC: X(I)*X(I)=2 (diagonal), X(I)*X(I+1)=-1 → 0.5*2*x_i^2 - 0.5*1*x_i*x_{i+1}
# Linear constraints C(I) for I=1..M=N/2: x(I) + x(M+I) >= 1
function qpband_model(N = 50000; T = Float64, backend = nothing, kwargs...)
    M = N ÷ 2
    RN = T(N)

    lvar = zeros(T, N)
    uvar = fill(T(2.0), N)
    start = zeros(T, N)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Objective linear: sum_i (-i/N)*x_i
    ExaModels.@add_obj(c, (-T(i) / RN) * x[i] for i in 1:N)

    # QUADRATIC section: X(I)*X(I)=2 → 0.5*2*x_i^2 = x_i^2
    #                    X(I)*X(I+1)=-1 → 0.5*(-1)*x_i*x_{i+1} ... wait
    # SIF QUADRATIC: "X X(I) X(I) 2.0  X(I+1) -1.0" means:
    #   H(x_i, x_i) += 2.0 → 0.5*2*x_i^2 = x_i^2
    #   H(x_i, x_{i+1}) += -1.0 → 0.5*(-1)*x_i*x_{i+1} = -0.5*x_i*x_{i+1}
    # Last row: X(N)*X(N)=2 → x_N^2
    ExaModels.@add_obj(c, x[i]^2 for i in 1:N)
    ExaModels.@add_obj(c, T(-0.5) * x[i] * x[i+1] for i in 1:N-1)

    # Constraints C(I): x(I) + x(M+I) >= 1
    ExaModels.@add_con(c, x[i] + x[M+i] - T(1.0) for i in 1:M;
                       lcon = T(0.0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
