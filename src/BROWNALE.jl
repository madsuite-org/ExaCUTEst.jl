# BROWNALE: Brown almost-linear system of nonlinear equations (N=200 default)
# Source: More', Garbow, Hillstrom (1981), problem 27; Buckley#79
# SIF classification: NOR2-AN-V-0 (nonlinear equations, no objective)
#
# N equality constraints (lcon=ucon=0):
#   c_I(x) = (sum_{J=1}^{N} x_J) + x_I - (N+1)   for I = 1..N-1
#   c_N(x) = x_1 * x_2 * ... * x_10 - 1
#
# SIF groups:
#   G(I) for I=1..N-1 (XE, identity type): v = sum_J x_J + x_I - (N+1)
#     = (all J with coeff 1.0) + extra coeff 1.0 on X(I) - RN+1
#   G(N) (XE, identity type): v = prod(x_1..x_10) - 1
#     element PROD = V1*V2*V3*V4*V5*V6*V7*V8*V9*V10 (first 10 variables)
#     constant RHS = 1.0
#
# All variables free (FR DEFAULT), start = 0.5 for all
function brownale_model(N = 200; T = Float64, backend = nothing, kwargs...)
    RNp1 = T(N + 1)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))

    # Constraints c_I(x) = 0 for I=1..N-1:
    #   v = (sum_{J=1}^{N} x_J) + x_I - (N+1)
    # CUTEst residual form: constraint = v - 0 = 0
    ExaModels.@add_con(
        c,
        (sum(x[j] for j in 1:N) + x[i] - RNp1)
        for i in 1:N-1
    )

    # Constraint c_N(x) = 0:
    #   v = x_1*x_2*...*x_10 - 1
    ExaModels.@add_con(c, x[1]*x[2]*x[3]*x[4]*x[5]*x[6]*x[7]*x[8]*x[9]*x[10] - T(1))

    return ExaModels.ExaModel(c; kwargs...)
end
