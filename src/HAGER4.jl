# HAGER4: Hager nonlinear optimal control problem P4
# N=2500, variables X(0..N) and U(1..N)
# Total: N+1 state + N control = 3N+1 = 7501 variables
# Objective: sum_{I=1}^N [ELT(X(I),X(I-1)) + (H/2)*U(I)^2]
# Equality constraints S(I): (1/H-1)*X(I) - (1/H)*X(I-1) - exp(T(I))*U(I) = 0
# X(0) fixed to XX0 = (1+3e)/(2-2e) where e=exp(1)
# U(I) bounded above by 1
#
# ELT element: F = D*(x-y)^2 (simplified... actually:
# ELT: F = D*X^2 + E*X*(Y-X) + F*(Y-X)^2
# where X=X(I), Y=X(I-1) from SIF ZV EL(I) X X(I), ZV EL(I) Y X(I-1)
# Parameters per element I:
#   DD = SCDA * Z(I-1), EE = SCDB * Z(I-1), FF = SCDC * Z(I-1)
#   Z(I) = exp(-2*T(I)), T(I) = I*H
#
# SCDA = 0.5 * (A(1) - A(0)) where A(I) = -0.5 * Z(I)
# SCDB = ((B(1)-B(0)) / H) where B(I) = A(I) * (T(I) + 0.5)
# SCDC = ((C(1)-C(0)) / (0.5*H^2)) where C(I) = A(I) * (T(I)^2 + T(I) + 0.5)
#
# But only elements for I=1..N use ELT, and I=1..2 only have SCA/SDB/SDC via
# DO I 0 1 (i.e. I=0 and I=1). But in the DO I 1 N loop, DD=SCDA*Z(I-1), etc.
# Actually SCDA, SCDB, SCDC are computed ONCE from I=0,1 data and used as
# scalar multipliers for Z(I-1) per element. So each element EL(I) has:
#   D_I = SCDA * Z(I-1), E_I = SCDB * Z(I-1), F_I = SCDC * Z(I-1)
#
# Let's compute SCDA, SCDB, SCDC:
# H = 1/N, T(0)=0, T(1)=H
# A(0) = -0.5*exp(0) = -0.5
# A(1) = -0.5*exp(-2H)
# DA = A(1) - A(0), SCDA = 0.5*DA
# B(0) = A(0)*(T(0)+0.5) = -0.5*0.5 = -0.25
# B(1) = A(1)*(T(1)+0.5)
# DB = B(1) - B(0), SCDB = DB/H
# C(0) = A(0)*(T(0)^2+T(0)+0.5) = -0.5*0.5 = -0.25
# C(1) = A(1)*(T(1)^2+T(1)+0.5)
# DC = C(1) - C(0), SCDC = DC/(0.5*H^2)
function hager4_model(N = 2500; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(N)
    H_half = H / 2

    # Compute SCDA, SCDB, SCDC
    A0 = -T(0.5) * exp(T(0))
    A1 = -T(0.5) * exp(-T(2) * H)
    DA = A1 - A0
    SCDA = T(0.5) * DA

    B0 = A0 * (T(0) + T(0.5))
    B1 = A1 * (H + T(0.5))
    DB = B1 - B0
    SCDB = DB / H

    C0 = A0 * (T(0)^2 + T(0) + T(0.5))
    C1 = A1 * (H^2 + H + T(0.5))
    DC = C1 - C0
    SCDC = DC / (T(0.5) * H^2)

    # Precompute Z(I-1) = exp(-2*T(I-1)) and exp(T(I)) for I=1..N
    ZI_vals = T[exp(-T(2) * H * (i - 1)) for i in 1:N]     # Z(I-1)
    ETI_vals = T[exp(H * i) for i in 1:N]                   # exp(T(I))

    # Initial state XX0
    e_val = exp(T(1))
    XX0 = (T(1) + T(3) * e_val) / (T(2) - T(2) * e_val)

    # Variables: x[1..N+1] = X(0)..X(N), x[N+2..2N+1] = U(1)..U(N)
    nvar = 2 * N + 1
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)

    # Fix X(0)
    lvar[1] = XX0
    uvar[1] = XX0
    start[1] = XX0

    # U(I) upper bound 1
    for i in 1:N
        uvar[N + 1 + i] = T(1)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_par(c, ZI, ZI_vals)
    ExaModels.@add_par(c, ETI, ETI_vals)

    # Objective: sum_{I=1}^N [ELT(x[I+1], x[I]) + H/2 * U(I)^2]
    # ELT: F = D*X^2 + E*X*(Y-X) + F*(Y-X)^2 where X=x[I+1], Y=x[I]
    # D = SCDA*Z(I-1), E = SCDB*Z(I-1), F = SCDC*Z(I-1)
    ExaModels.@add_obj(c,
        (SCDA * ZI[i]) * x[i + 1]^2 +
        (SCDB * ZI[i]) * x[i + 1] * (x[i] - x[i + 1]) +
        (SCDC * ZI[i]) * (x[i] - x[i + 1])^2 +
        H_half * x[N + 1 + i]^2
        for i in 1:N)

    # Equality constraints S(I): (1/H-1)*X(I) - (1/H)*X(I-1) - exp(T(I))*U(I) = 0
    # i.e., (1/H-1)*x[I+1] - (1/H)*x[I] - ETI[I]*x[N+1+I] = 0
    one_over_H = T(1) / H
    one_over_H_minus_1 = one_over_H - T(1)
    ExaModels.@add_con(c,
        one_over_H_minus_1 * x[i + 1] + (-one_over_H) * x[i] - ETI[i] * x[N + 1 + i]
        for i in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
