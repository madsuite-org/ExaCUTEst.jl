# RK23: Embedded Runge-Kutta pair optimization, fixed-size
# Find coefficients for embedded 2nd/3rd order RK pair minimizing LTE
# Source: Hairer, Norsett, Wanner (1980); SIF input by S. Leyffer, 1997
# 17 variables, 11 constraints
#
# Variable ordering: C2=1, A21=2, C3=3, A31=4, A32=5, B1=6, B2=7, B3=8,
#                   BB1=9, BB2=10, BB3=11, TP1=12, TM1=13, TP2=14, TM2=15, TP3=16, TM3=17
#
# Objective (N type, linear): TP1+TM1+TP2+TM2+TP3+TM3
#
# Constraints (E type, equality = 0):
#   ROWS1:   A21 - C2 = 0
#   ROWS2:   A31 + A32 - C3 = 0
#   FIRST2:  B1 + B2 + B3 - 1 = 0
#   FIRST3:  BB1 + BB2 + BB3 - 1 = 0
#   SECND2:  B2*C2 + B3*C3 - 0.5 = 0
#   SECND3:  BB2*C2 + BB3*C3 - 0.5 = 0
#   THIRD31: BB2*C2^2 + BB3*C3^2 - 1/3 = 0
#   THIRD32: BB3*A32*C2 - 1/6 = 0
#   ART1:    TP1 - TM2 + 4*BB2*C2^3 + 4*BB3*C3^3 - 1 = 0
#   ART2:    TP2 - TM2 + 8*BB3*C3*A32*C2 - 1 = 0
#   ART3:    TP3 - TM3 + 12*BB3*A32*C2^2 - 1 = 0
#
# Bounds: all free; TP1,TM1,TP2,TM2,TP3,TM3 >= 0
function rk23_model(; T = Float64, backend = nothing, kwargs...)
    ONETHIRD  = T(1) / T(3)
    ONESIXTH  = T(1) / T(6)

    # Variable ordering: C2=1, A21=2, C3=3, A31=4, A32=5, B1=6, B2=7, B3=8,
    #                   BB1=9, BB2=10, BB3=11, TP1=12, TM1=13, TP2=14, TM2=15, TP3=16, TM3=17
    lvar = fill(T(-Inf), 17)
    uvar = fill(T(Inf), 17)
    # TP1=12, TM1=13, TP2=14, TM2=15, TP3=16, TM3=17 >= 0
    for i in 12:17
        lvar[i] = T(0)
    end

    start = zeros(T, 17)
    # Classical RK23 start point
    start[1]  = T(1.0)   # C2
    start[2]  = T(1.0)   # A21
    start[3]  = T(0.5)   # C3
    start[4]  = T(0.25)  # A31
    start[5]  = T(0.25)  # A32
    start[6]  = T(0.5)   # B1
    start[7]  = T(0.5)   # B2
    start[8]  = T(0.0)   # B3
    start[9]  = T(1) / T(6)   # BB1
    start[10] = T(1) / T(6)   # BB2
    start[11] = T(4) / T(6)   # BB3

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 17; lvar = lvar, uvar = uvar, start = start)

    # Objective: TP1+TM1+TP2+TM2+TP3+TM3
    ExaModels.@add_obj(c, x[12] + x[13] + x[14] + x[15] + x[16] + x[17])

    # Constraint ROWS1: A21 - C2 = 0
    ExaModels.@add_con(c, x[2] - x[1])
    # Constraint ROWS2: A31 + A32 - C3 = 0
    ExaModels.@add_con(c, x[4] + x[5] - x[3])
    # Constraint FIRST2: B1 + B2 + B3 - 1 = 0
    ExaModels.@add_con(c, x[6] + x[7] + x[8] - T(1))
    # Constraint FIRST3: BB1 + BB2 + BB3 - 1 = 0
    ExaModels.@add_con(c, x[9] + x[10] + x[11] - T(1))
    # Constraint SECND2: B2*C2 + B3*C3 - 0.5 = 0
    ExaModels.@add_con(c, x[7] * x[1] + x[8] * x[3] - T(0.5))
    # Constraint SECND3: BB2*C2 + BB3*C3 - 0.5 = 0
    ExaModels.@add_con(c, x[10] * x[1] + x[11] * x[3] - T(0.5))
    # Constraint THIRD31: BB2*C2^2 + BB3*C3^2 - 1/3 = 0
    ExaModels.@add_con(c, x[10] * x[1]^2 + x[11] * x[3]^2 - ONETHIRD)
    # Constraint THIRD32: BB3*A32*C2 - 1/6 = 0
    ExaModels.@add_con(c, x[11] * x[5] * x[1] - ONESIXTH)
    # Constraint ART1: TP1 - TM2 + 4*BB2*C2^3 + 4*BB3*C3^3 - 1 = 0
    ExaModels.@add_con(c, x[12] - x[15] + T(4) * x[10] * x[1]^3 + T(4) * x[11] * x[3]^3 - T(1))
    # Constraint ART2: TP2 - TM2 + 8*BB3*C3*A32*C2 - 1 = 0
    ExaModels.@add_con(c, x[14] - x[15] + T(8) * x[11] * x[3] * x[5] * x[1] - T(1))
    # Constraint ART3: TP3 - TM3 + 12*BB3*A32*C2^2 - 1 = 0
    ExaModels.@add_con(c, x[16] - x[17] + T(12) * x[11] * x[5] * x[1]^2 - T(1))

    return ExaModels.ExaModel(c; kwargs...)
end
