# ALLINQP: Banded QP with all sorts of constraint and variable bounds
# N=50000 variables (default), M=N/2 constraints of E/G/L types
# Objective: linear terms -I/N for I in 3N/7+1..4N/7, plus QUADRATIC section
# QUADRATIC: 0.5 * 2 * x[I]^2 + 0.5 * (-1) * x[I] * x[I+1] for I=1..N-1, + 0.5*2*x[N]^2
#   = x[I]^2 - 0.5 * x[I] * x[I+1]
# Constraints: C(I) = x[I] + x[M+I] - 1
#   I=1..M/4: equality (= 0)
#   I=M/4+1..2M/4: G type (≥ 0), no range
#   I=2M/4+1..3M/4: G type with RANGE 1.0 → 0 ≤ C(I) ≤ 1
#   I=3M/4+1..M: L type (≤ 0)
# Bounds: default FR, with partial override sections
# Source: Nick Gould, December 1999
function allinqp_model(N = 50000; T = Float64, backend = nothing, kwargs...)
    M     = N ÷ 2
    M4    = M ÷ 4
    M4_1  = M4 + 1
    M2_4  = 2 * M4
    M2_4_1 = M2_4 + 1
    M3_4  = 3 * M4
    M3_4_1 = M3_4 + 1
    N7    = N ÷ 7
    N7_1  = N7 + 1
    N2_7  = 2 * N7
    N2_7_1 = N2_7 + 1
    N3_7  = 3 * N7
    N3_7_1 = N3_7 + 1
    N4_7  = 4 * N7
    N4_7_1 = N4_7 + 1
    N5_7  = 5 * N7
    N5_7_1 = N5_7 + 1
    N6_7  = 6 * N7
    N6_7_1 = N6_7 + 1

    # Build variable bounds
    # Default: FR (free, lvar=-Inf, uvar=Inf)
    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    # N/7+1..2N/7: XL 0.0
    for i in N7_1:N2_7
        lvar[i] = T(0)
    end
    # 2N/7+1..3N/7: XL 0.0
    for i in N2_7_1:N3_7
        lvar[i] = T(0)
    end
    # 3N/7+1..4N/7: XL 0.5, XU 2.0
    for i in N3_7_1:N4_7
        lvar[i] = T(0.5)
        uvar[i] = T(2.0)
    end
    # 4N/7+1..5N/7: XU 3.0
    for i in N4_7_1:N5_7
        uvar[i] = T(3.0)
    end
    # 5N/7+1..6N/7: XU 0.0
    for i in N5_7_1:N6_7
        uvar[i] = T(0.0)
    end
    # 6N/7+1..N: XX 0.0 (fixed)
    for i in N6_7_1:N
        lvar[i] = T(0)
        uvar[i] = T(0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar)

    # Objective: linear terms for I=3N/7+1..4N/7
    rN = T(N)
    obj_data = [(i = i, ri = T(i) / rN) for i in N3_7_1:N4_7]
    ExaModels.@add_obj(c, T(-1) * d.ri * x[d.i] for d in obj_data)

    # QUADRATIC: X(I) X(I) 2.0 → 0.5*2*x[I]^2 = x[I]^2   for I=1..N-1
    #            X(I) X(I+1) -1.0 → 0.5*(-1)*x[I]*x[I+1]  for I=1..N-1
    #            X(N) X(N) 2.0 → x[N]^2
    quad_data = [(i = i,) for i in 1:N-1]
    ExaModels.@add_obj(c, x[d.i]^2 - T(0.5) * x[d.i] * x[d.i + 1] for d in quad_data)
    ExaModels.@add_obj(c, x[N]^2)

    # Constraints: C(I) = x[I] + x[M+I] - 1
    # I=1..M/4: equality
    eq_data = [(i = i,) for i in 1:M4]
    ExaModels.@add_con(c, x[d.i] + x[M + d.i] - T(1) for d in eq_data)

    # I=M/4+1..2M/4: G type (≥ 0), lcon=0, ucon=Inf
    g1_data = [(i = i,) for i in M4_1:M2_4]
    ExaModels.@add_con(c, x[d.i] + x[M + d.i] - T(1)
        for d in g1_data; lcon = T(0), ucon = T(Inf))

    # I=2M/4+1..3M/4: G type with RANGE 1.0 → 0 ≤ C ≤ 1
    g2_data = [(i = i,) for i in M2_4_1:M3_4]
    ExaModels.@add_con(c, x[d.i] + x[M + d.i] - T(1)
        for d in g2_data; lcon = T(0), ucon = T(1))

    # I=3M/4+1..M: L type (≤ 0), lcon=-Inf, ucon=0
    l_data = [(i = i,) for i in M3_4_1:M]
    ExaModels.@add_con(c, x[d.i] + x[M + d.i] - T(1)
        for d in l_data; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
