# GOULDQP1: QP problem class 1 (Hock-Schittkowski 118 modified), 32 vars, 17 equality constraints
# Variables: X(1..15), AS(1..4), CS(1..4), BS(1..4), DS(1..5)
# Linear objective + quadratic terms in X(1..15)
function gouldqp1_model(; T = Float64, backend = nothing, kwargs...)
    NV = 32
    # Variable indices
    X(i)  = i          # i = 1..15
    AS(k) = 15 + k     # k = 1..4 → 16..19
    CS(k) = 19 + k     # k = 1..4 → 20..23
    BS(k) = 23 + k     # k = 1..4 → 24..27
    DS(k) = 27 + k     # k = 1..5 → 28..32

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)

    # Bounds for X(1..15)
    # X(1): [8, 21]
    lvar[X(1)] = T(8); uvar[X(1)] = T(21)
    # X(2): [43, 57]
    lvar[X(2)] = T(43); uvar[X(2)] = T(57)
    # X(3): [3, 16]
    lvar[X(3)] = T(3); uvar[X(3)] = T(16)
    # X(3K+1) for K=1..4: upper=90
    for k in 1:4
        uvar[X(3 * k + 1)] = T(90)
    end
    # X(3K+2) for K=1..4: upper=120
    for k in 1:4
        uvar[X(3 * k + 2)] = T(120)
    end
    # X(3K+3) for K=1..4: upper=60
    for k in 1:4
        uvar[X(3 * k + 3)] = T(60)
    end
    # AS,BS: [0,13]
    for k in 1:4
        lvar[AS(k)] = T(0); uvar[AS(k)] = T(13)
        lvar[BS(k)] = T(0); uvar[BS(k)] = T(13)
    end
    # CS: [0,14]
    for k in 1:4
        lvar[CS(k)] = T(0); uvar[CS(k)] = T(14)
    end
    # DS(k) bounds
    ds_lower = T[0, 0, 0, 0, 0]
    ds_upper = T[60, 50, 70, 85, 100]
    for k in 1:5
        lvar[DS(k)] = ds_lower[k]; uvar[DS(k)] = ds_upper[k]
    end

    # Starting point
    x0 = fill(T(20), NV)
    x0[X(2)] = T(55); x0[X(3)] = T(15)
    x0[X(5)] = T(60); x0[X(8)] = T(60); x0[X(11)] = T(60); x0[X(14)] = T(60)
    for k in 1:4
        x0[AS(k)] = T(7); x0[BS(k)] = T(7); x0[CS(k)] = T(7)
    end
    x0[BS(1)] = T(12); x0[CS(1)] = T(12)
    x0[DS(1)] = T(30); x0[DS(2)] = T(50); x0[DS(3)] = T(30); x0[DS(4)] = T(15); x0[DS(5)] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: linear in X(K*3+1,K*3+2,K*3+3) + quadratic via SQ(X(I))
    # Linear part: sum_{K=0}^{4} [2.3*X(3K+1) + 1.7*X(3K+2) + 2.2*X(3K+3)]
    # Quadratic coefficients on X(1..15): see GROUP USES
    quad_coeffs = T[-1, 0.0001, 0.00015, -0.0001, 0.0001, 10, -0.0001, 0.0001, 25, -2.5, 0.0001, 0.00015, -0.0001, 0.0001, 0.00015]

    # Linear objective
    ExaModels.@add_obj(c,
        sum(T(2.3) * x[X(3 * k + 1)] + T(1.7) * x[X(3 * k + 2)] + T(2.2) * x[X(3 * k + 3)] for k in 0:4))

    # Quadratic objective terms: quad_coeffs[I] * X(I)^2
    quad_data = [(i = i, c = quad_coeffs[i]) for i in 1:15]
    ExaModels.@add_obj(c,
        d.c * x[d.i]^2
        for d in quad_data)

    # A(K) constraints: X(3K+1) - X(3K-2) - AS(K) - 7 = 0, for K=1..4
    # (rhs=7 from constants A(K)=-7, so value = expr - (-7) = expr + 7)
    # Wait: constants A(K)=-7, so group value = linear - rhs = X(3K+1)-X(3K-2)-AS(K) - (-7)
    # = X(3K+1) - X(3K-2) - AS(K) + 7 = 0
    for k in 1:4
        ExaModels.@add_con(c,
            x[X(3 * k + 1)] - x[X(3 * k - 2)] - x[AS(k)] + T(7))
    end

    # B(K) constraints: X(3K+3) - X(3K) - BS(K) + 7 = 0, for K=1..4
    for k in 1:4
        ExaModels.@add_con(c,
            x[X(3 * k + 3)] - x[X(3 * k)] - x[BS(k)] + T(7))
    end

    # C(K) constraints: X(3K+2) - X(3K-1) - CS(K) + 7 = 0, for K=1..4
    for k in 1:4
        ExaModels.@add_con(c,
            x[X(3 * k + 2)] - x[X(3 * k - 1)] - x[CS(k)] + T(7))
    end

    # D constraints:
    # D1: X1+X2+X3 - DS1 - 60 = 0
    ExaModels.@add_con(c, x[X(1)] + x[X(2)] + x[X(3)] - x[DS(1)] - T(60))
    # D2: X4+X5+X6 - DS2 - 50 = 0
    ExaModels.@add_con(c, x[X(4)] + x[X(5)] + x[X(6)] - x[DS(2)] - T(50))
    # D3: X7+X8+X9 - DS3 - 70 = 0
    ExaModels.@add_con(c, x[X(7)] + x[X(8)] + x[X(9)] - x[DS(3)] - T(70))
    # D4: X10+X11+X12 - DS4 - 85 = 0
    ExaModels.@add_con(c, x[X(10)] + x[X(11)] + x[X(12)] - x[DS(4)] - T(85))
    # D5: X13+X14+X15 - DS5 - 100 = 0
    ExaModels.@add_con(c, x[X(13)] + x[X(14)] + x[X(15)] - x[DS(5)] - T(100))

    return ExaModels.ExaModel(c; kwargs...)
end
