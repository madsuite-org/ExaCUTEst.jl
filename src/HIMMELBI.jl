# HIMMELBI: Weapon assignment problem (Bracken & McCormick / Himmelblau problem 23)
# 100 variables X(I,J) for weapon type I=1..5 and target J=1..20
# Variable ordering: X(1,1), X(2,1), ..., X(5,1), X(1,2), ..., X(5,20)
# (inner loop over I, outer loop over J → index = (J-1)*5 + I)
# All variables in [0, NW] where NW = sum C(I) = 1000.0
#
# Objective: sum_J U(J) * (prod_I A(I,J)^x[(J-1)*5+I] - 1)
# derived from: group P(J) with identity type, SCALE = 1/U(J), CONSTANT = 1
# element 5PEXP = A1^Y1 * A2^Y2 * A3^Y3 * A4^Y4 * A5^Y5
# contribution = v / sigma = (element - 1) * U(J)
#
# Constraints:
# CB(J) for J=1,6,10,14,15,16,20: sum_I x[(J-1)*5+I] - B(J) >= 0  (G type, lcon=0)
# CC(I) for I=1..5:              sum_J x[(J-1)*5+I] - C(I) <= 0  (L type, ucon=0)

function himmelbi_model(; T = Float64, backend = nothing, kwargs...)
    # A(I,J): probability that weapon I will damage target J
    # Row = target J (1..20), Col = weapon type I (1..5)
    A = T[
        # J=1
        1.0,  0.84, 0.96, 1.0,  0.92,
        # J=2
        0.95, 0.83, 0.95, 1.0,  0.94,
        # J=3
        1.0,  0.85, 0.95, 1.0,  0.92,
        # J=4
        1.0,  0.84, 0.96, 1.0,  0.95,
        # J=5
        1.0,  0.85, 0.96, 1.0,  0.95,
        # J=6
        0.85, 0.81, 0.90, 1.0,  0.98,
        # J=7
        0.90, 0.81, 0.92, 1.0,  0.98,
        # J=8
        0.85, 0.82, 0.91, 1.0,  1.0,
        # J=9
        0.80, 0.80, 0.92, 1.0,  1.0,
        # J=10
        1.0,  0.86, 0.95, 0.96, 0.90,
        # J=11
        1.0,  1.0,  0.99, 0.91, 0.95,
        # J=12
        1.0,  0.98, 0.98, 0.92, 0.96,
        # J=13
        1.0,  1.0,  0.99, 0.91, 0.91,
        # J=14
        1.0,  0.88, 0.98, 0.92, 0.98,
        # J=15
        1.0,  0.87, 0.97, 0.98, 0.99,
        # J=16
        1.0,  0.88, 0.98, 0.93, 0.99,
        # J=17
        1.0,  0.85, 0.95, 1.0,  1.0,
        # J=18
        0.95, 0.84, 0.92, 1.0,  1.0,
        # J=19
        1.0,  0.85, 0.93, 1.0,  1.0,
        # J=20
        1.0,  0.85, 0.92, 1.0,  1.0,
    ]
    # U(J): military value of target J
    U = T[60., 50., 50., 75., 40., 60., 35., 30., 25., 150.,
          30., 45., 125., 200., 200., 130., 100., 100., 100., 150.]
    # C(I): number of available weapons of type I
    C = T[200.0, 100.0, 300.0, 150.0, 250.0]
    # NW = sum C(I) = 1000.0
    NW = T(1000.0)
    # B(J): minimum number of weapons for selected targets
    B = Dict(1 => T(30.0), 6 => T(100.0), 10 => T(40.0), 14 => T(50.0),
             15 => T(70.0), 16 => T(35.0), 20 => T(10.0))

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # 100 variables X(I,J), all in [0, NW], start at 0
    ExaModels.@add_var(c, x, 100;
        lvar  = fill(T(0.0), 100),
        uvar  = fill(NW, 100),
        start = fill(T(0.0), 100))

    # Objective: sum_J U(J) * (A(1,J)^x[(J-1)*5+1] * ... * A(5,J)^x[(J-1)*5+5] - 1)
    # Store A and U as parameters for the loop
    ExaModels.@add_par(c, Ap, A)
    ExaModels.@add_par(c, Up, U)
    ExaModels.@add_obj(c,
        Up[j] * (Ap[(j-1)*5+1]^x[(j-1)*5+1] *
                 Ap[(j-1)*5+2]^x[(j-1)*5+2] *
                 Ap[(j-1)*5+3]^x[(j-1)*5+3] *
                 Ap[(j-1)*5+4]^x[(j-1)*5+4] *
                 Ap[(j-1)*5+5]^x[(j-1)*5+5] - T(1))
        for j in 1:20)

    # CB constraints: G type (>= 0), lcon=0, ucon=Inf
    # sum_I x[(J-1)*5+I] - B(J) >= 0  ↔  constraint value = sum_I x - B(J), lcon=0
    for (jidx, j) in enumerate([1, 6, 10, 14, 15, 16, 20])
        bj = B[j]
        base = (j - 1) * 5
        ExaModels.@add_con(c,
            x[base+1] + x[base+2] + x[base+3] + x[base+4] + x[base+5] - bj;
            lcon = T(0), ucon = T(Inf))
    end

    # CC constraints: L type (<= 0), lcon=-Inf, ucon=0
    # sum_J x[(J-1)*5+I] - C(I) <= 0  ↔  constraint value = sum_J x - C(I), ucon=0
    ExaModels.@add_par(c, Cp, C)
    ExaModels.@add_con(c,
        sum(x[(j-1)*5+i] for j in 1:20) - Cp[i]
        for i in 1:5;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
