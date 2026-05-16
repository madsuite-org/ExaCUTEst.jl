# ELATTAR: Nonlinear minmax problem in 7 variables (6 + auxiliary U)
# Minimize U subject to:
#   ET1(I) + ET2(I) - Y(I) - U <= 0  [F(I) <= 0, L-type]
#   -ET1(I) - ET2(I) + Y(I) - U <= 0  [MF(I) <= 0, L-type]
# ET1(I) = X1 * exp(-X2*T(I)) * cos(X3*T(I) + X4)
# ET2(I) = X5 * exp(-X6*T(I))
# T(I) = (I-1)*0.1 for I=1..51
# Variables: X1..X6 (indices 1-6), U (index 7); all free
# Start: X1=-2, X2=-2, X3=7, X4=0, X5=-2, X6=1, U=0
function elattar_model(; T = Float64, backend = nothing, kwargs...)
    NI = 51
    NV = 7  # X1..X6, U

    # T(I) = (I-1)*0.1
    ti = [(i - 1) * T(0.1) for i in 1:NI]

    # Y(I) = 0.5*exp(T) - exp(-2T) + 0.5*exp(-3T) + 1.5*exp(-3T/2)*sin(7T) + exp(-5T/2)*sin(5T)
    yi = [T(0.5) * exp(ti[i]) - exp(-2 * ti[i]) + T(0.5) * exp(-3 * ti[i]) +
          T(1.5) * exp(-3 * ti[i] / 2) * sin(7 * ti[i]) + exp(-5 * ti[i] / 2) * sin(5 * ti[i])
          for i in 1:NI]

    x0 = zeros(T, NV)
    x0[1] = T(-2)  # X1
    x0[2] = T(-2)  # X2
    x0[3] = T(7)   # X3
    x0[4] = T(0)   # X4
    x0[5] = T(-2)  # X5
    x0[6] = T(1)   # X6
    x0[7] = T(0)   # U

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective: minimize U
    ExaModels.@add_obj(c, x[7])

    # F(I) constraints: ET1(I) + ET2(I) - Y(I) <= U, i.e., ET1+ET2 - yi - U <= 0
    # MF(I) constraints: -(ET1(I) + ET2(I)) + Y(I) <= U, i.e., -ET1-ET2 + yi - U <= 0
    con_data = [(ti = ti[i], yi = yi[i]) for i in 1:NI]

    # ET1 = X1 * exp(-X2 * T) * cos(X3 * T + X4)
    # ET2 = X5 * exp(-X6 * T)
    # F(I): ET1 + ET2 - yi - U <= 0 → lcon=-Inf, ucon=0
    ExaModels.@add_con(c,
        x[1] * exp(-x[2] * d.ti) * cos(x[3] * d.ti + x[4]) +
        x[5] * exp(-x[6] * d.ti) - d.yi - x[7]
        for d in con_data;
        lcon = T(-Inf), ucon = T(0))

    # MF(I): -(ET1 + ET2) + yi - U <= 0 → lcon=-Inf, ucon=0
    ExaModels.@add_con(c,
        -(x[1] * exp(-x[2] * d.ti) * cos(x[3] * d.ti + x[4]) +
          x[5] * exp(-x[6] * d.ti)) + d.yi - x[7]
        for d in con_data;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
