# LOADBAL: static load balancing in tree computer network
# 11 nodes, 20 directed links, 31 variables
# Variables: 20 link flows X + 11 processing times B(1..11)
# Objective: sum_I B(I)/(CB_I-B(I)) + sum_I [V/(CIJ-(A*V+B*W)) for GA/GB groups]
#   (rational functions, no group type means identity: contribution = element value)
# Constraints: N(I) conservation equality: sum of incident flows - B(I) = RHS_I
#              CNST(I) capacity: 20*V + 80*W <= CIJE (linear, L type)
# Default bounds: >= 0; upper bounds on B(i)
# Variable order in SIF: X4,1 X1,4 X4,2 X2,4 X4,3 X3,4 X4,5 X5,4 X5,6 X6,5
#                        X5,7 X7,5 X5,8 X8,5 X8,9 X9,8 X8,10 X10,8 X8,11 X11,8
#                        B1..B11
function loadbal_model(; T = Float64, backend = nothing, kwargs...)
    # FI = 514.0
    # A=80, B=20 (from GLOBALS)
    # CIJ = 1000 (COMA1/COMB1) or 10000 (COMA2/COMB2)

    # Variable indices (1-based)
    # Link flows: 1=X4,1 2=X1,4 3=X4,2 4=X2,4 5=X4,3 6=X3,4 7=X4,5 8=X5,4
    #             9=X5,6 10=X6,5 11=X5,7 12=X7,5 13=X5,8 14=X8,5
    #             15=X8,9 16=X9,8 17=X8,10 18=X10,8 19=X8,11 20=X11,8
    # Processing times: 21=B1 22=B2 23=B3 24=B4 25=B5 26=B6 27=B7 28=B8 29=B9 30=B10 31=B11

    lvar = fill(T(0), 31)  # default >= 0
    uvar = fill(T(Inf), 31)
    # Upper bounds on B(i)
    uvar[21] = T(99.99)  # B1
    uvar[22] = T(99.99)  # B2
    uvar[23] = T(19.99)  # B3
    uvar[24] = T(99.99)  # B4
    uvar[25] = T(99.99)  # B5
    uvar[26] = T(19.99)  # B6
    uvar[27] = T(19.99)  # B7
    uvar[28] = T(99.99)  # B8
    uvar[29] = T(19.99)  # B9
    uvar[30] = T(19.99)  # B10
    uvar[31] = T(19.99)  # B11

    start = zeros(T, 31)
    start[1]  = T(0.0)   # X4,1
    start[2]  = T(0.0)   # X1,4
    start[3]  = T(0.0)   # X4,2
    start[4]  = T(0.0)   # X2,4
    start[5]  = T(0.0)   # X4,3
    start[6]  = T(0.0)   # X3,4
    start[7]  = T(0.0)   # X4,5
    start[8]  = T(0.0)   # X5,4
    start[9]  = T(0.0)   # X5,6
    start[10] = T(0.0)   # X6,5
    start[11] = T(0.0)   # X5,7
    start[12] = T(0.0)   # X7,5
    start[13] = T(0.0)   # X5,8
    start[14] = T(0.0)   # X8,5
    start[15] = T(0.0)   # X8,9
    start[16] = T(0.0)   # X9,8
    start[17] = T(0.0)   # X8,10
    start[18] = T(0.0)   # X10,8
    start[19] = T(0.0)   # X8,11
    start[20] = T(0.0)   # X11,8
    start[21] = T(95.0)  # B1
    start[22] = T(95.0)  # B2
    start[23] = T(19.0)  # B3
    start[24] = T(70.0)  # B4
    start[25] = T(70.0)  # B5
    start[26] = T(19.0)  # B6
    start[27] = T(19.0)  # B7
    start[28] = T(70.0)  # B8
    start[29] = T(19.0)  # B9
    start[30] = T(19.0)  # B10
    start[31] = T(19.0)  # B11

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 31; lvar = lvar, uvar = uvar, start = start)

    # Objective: F(I) groups - BETA elements (identity group type, scale ignored)
    # BETA1: CB=100, F = V/(CB-V) for nodes 1,2,4,5,8 -> B1,B2,B4,B5,B8
    # BETA2: CB=20,  F = V/(CB-V) for nodes 3,6,7,9,10,11 -> B3,B6,B7,B9,B10,B11
    ExaModels.@add_obj(c, x[21] / (T(100) - x[21]))   # B1
    ExaModels.@add_obj(c, x[22] / (T(100) - x[22]))   # B2
    ExaModels.@add_obj(c, x[23] / (T(20)  - x[23]))   # B3
    ExaModels.@add_obj(c, x[24] / (T(100) - x[24]))   # B4
    ExaModels.@add_obj(c, x[25] / (T(100) - x[25]))   # B5
    ExaModels.@add_obj(c, x[26] / (T(20)  - x[26]))   # B6
    ExaModels.@add_obj(c, x[27] / (T(20)  - x[27]))   # B7
    ExaModels.@add_obj(c, x[28] / (T(100) - x[28]))   # B8
    ExaModels.@add_obj(c, x[29] / (T(20)  - x[29]))   # B9
    ExaModels.@add_obj(c, x[30] / (T(20)  - x[30]))   # B10
    ExaModels.@add_obj(c, x[31] / (T(20)  - x[31]))   # B11

    # GA(I) groups (N type, identity): COMA1/2 elements: V/(CIJ-(A*V+B*W)), A=80, B=20
    # COMA1 (CIJ=1000): V/(1000-(80*V+20*W))
    # EGA(1): V=X4,1(1), W=X1,4(2)
    # EGA(2): V=X4,2(3), W=X2,4(4)
    # EGA(3): V=X4,3(5), W=X3,4(6)
    # EGA(4): V=X1,4(2), W=X4,1(1)   [I+3=4, from loop I=1..3: I+3=4, EGA(I+3) uses X(I,4C) and X(4C,I)]
    # EGA(5): V=X2,4(4), W=X4,2(3)   [I+3=5]
    # EGA(6): V=X3,4(6), W=X4,3(5)   [I+3=6]
    # For I=1..3, I+6=7,8,9: EGA(I+6) uses V=X(8C,I+8), W=X(I+8,8C)
    # I=1: I+8=9 -> V=X8,9(15), W=X9,8(16)
    # I=2: I+8=10 -> V=X8,10(17), W=X10,8(18)
    # I=3: I+8=11 -> V=X8,11(19), W=X11,8(20)
    # For I=1..3, I+9=10,11,12: EGA(I+9) uses V=X(I+8,8C), W=X(8C,I+8)
    # I=1: V=X9,8(16), W=X8,9(15)
    # I=2: V=X10,8(18), W=X8,10(17)
    # I=3: V=X11,8(20), W=X8,11(19)
    # Loop I=6..7 (6C..7C): I2=2*I, EGA(I2+1), EGA(I2+2)
    # I=6: I2=12, EGA(13) uses V=X5,6(9), W=X6,5(10); EGA(14) uses V=X6,5(10), W=X5,6(9)
    # I=7: I2=14, EGA(15) uses V=X5,7(11), W=X7,5(12); EGA(16) uses V=X7,5(12), W=X5,7(11)
    # COMA2 (CIJ=10000): EGA(17..20)
    # EGA17: V=X5,4(8), W=X4,5(7)
    # EGA18: V=X4,5(7), W=X5,4(8)
    # EGA19: V=X5,8(13), W=X8,5(14)
    # EGA20: V=X8,5(14), W=X5,8(13)

    # GA groups 1-16 use COMA1 (CIJ=1000)
    # v_idx list for GA(1..16) elements: (v, w) pairs
    ga_data_1000 = [
        (v=1, w=2),   # GA1:  X4,1  X1,4
        (v=3, w=4),   # GA2:  X4,2  X2,4
        (v=5, w=6),   # GA3:  X4,3  X3,4
        (v=2, w=1),   # GA4:  X1,4  X4,1
        (v=4, w=3),   # GA5:  X2,4  X4,2
        (v=6, w=5),   # GA6:  X3,4  X4,3
        (v=15, w=16), # GA7:  X8,9  X9,8
        (v=17, w=18), # GA8:  X8,10 X10,8
        (v=19, w=20), # GA9:  X8,11 X11,8
        (v=16, w=15), # GA10: X9,8  X8,9
        (v=18, w=17), # GA11: X10,8 X8,10
        (v=20, w=19), # GA12: X11,8 X8,11
        (v=9,  w=10), # GA13: X5,6  X6,5
        (v=10, w=9),  # GA14: X6,5  X5,6
        (v=11, w=12), # GA15: X5,7  X7,5
        (v=12, w=11), # GA16: X7,5  X5,7
    ]
    ExaModels.@add_obj(c, x[d.v] / (T(1000) - (T(80)*x[d.v] + T(20)*x[d.w]))
                       for d in ga_data_1000)

    # GA groups 17-20 use COMA2 (CIJ=10000)
    ga_data_10000 = [
        (v=8,  w=7),  # GA17: X5,4  X4,5
        (v=7,  w=8),  # GA18: X4,5  X5,4
        (v=13, w=14), # GA19: X5,8  X8,5
        (v=14, w=13), # GA20: X8,5  X5,8
    ]
    ExaModels.@add_obj(c, x[d.v] / (T(10000) - (T(80)*x[d.v] + T(20)*x[d.w]))
                       for d in ga_data_10000)

    # GB(I) groups (N type, identity): COMB1/2: V/(CIJ-(B*V+A*W)), A=80, B=20
    # Same pairing but V/(CIJ-(20*V+80*W))
    ExaModels.@add_obj(c, x[d.v] / (T(1000) - (T(20)*x[d.v] + T(80)*x[d.w]))
                       for d in ga_data_1000)
    ExaModels.@add_obj(c, x[d.v] / (T(10000) - (T(20)*x[d.v] + T(80)*x[d.w]))
                       for d in ga_data_10000)

    # Conservation constraints N(I) (E type, equality): sum of flows - B(I) = RHS_I
    # From VARIABLES section: each X contributes ±1 to N(i), N(j)
    # N1: X4,1(1)*1 + X1,4(2)*(-1) - B1(21) = -95
    # N2: X4,2(3)*1 + X2,4(4)*(-1) - B2(22) = -95
    # N3: X4,3(5)*1 + X3,4(6)*(-1) - B3(23) = -19
    # N4: X4,1(1)*(-1) + X1,4(2)*1 + X4,2(3)*(-1) + X2,4(4)*1 + X4,3(5)*(-1) + X3,4(6)*1 + X4,5(7)*(-1) + X5,4(8)*1 - B4(24) = -70
    # N5: X4,5(7)*1 + X5,4(8)*(-1) + X5,6(9)*(-1) + X6,5(10)*1 + X5,7(11)*(-1) + X7,5(12)*1 + X5,8(13)*(-1) + X8,5(14)*1 - B5(25) = -70
    # N6: X5,6(9)*1 + X6,5(10)*(-1) - B6(26) = -19
    # N7: X5,7(11)*1 + X7,5(12)*(-1) - B7(27) = -19
    # N8: X5,8(13)*1 + X8,5(14)*(-1) + X8,9(15)*(-1) + X9,8(16)*1 + X8,10(17)*(-1) + X10,8(18)*1 + X8,11(19)*(-1) + X11,8(20)*1 - B8(28) = -70
    # N9: X8,9(15)*1 + X9,8(16)*(-1) - B9(29) = -19
    # N10: X8,10(17)*1 + X10,8(18)*(-1) - B10(30) = -19
    # N11: X8,11(19)*1 + X11,8(20)*(-1) - B11(31) = -19
    ExaModels.@add_con(c,  x[1] - x[2] - x[21] + T(95))   # N1
    ExaModels.@add_con(c,  x[3] - x[4] - x[22] + T(95))   # N2
    ExaModels.@add_con(c,  x[5] - x[6] - x[23] + T(19))   # N3
    ExaModels.@add_con(c, -x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[24] + T(70))  # N4
    ExaModels.@add_con(c,  x[7] - x[8] - x[9] + x[10] - x[11] + x[12] - x[13] + x[14] - x[25] + T(70))  # N5
    ExaModels.@add_con(c,  x[9] - x[10] - x[26] + T(19))   # N6
    ExaModels.@add_con(c,  x[11] - x[12] - x[27] + T(19))  # N7
    ExaModels.@add_con(c,  x[13] - x[14] - x[15] + x[16] - x[17] + x[18] - x[19] + x[20] - x[28] + T(70))  # N8
    ExaModels.@add_con(c,  x[15] - x[16] - x[29] + T(19))  # N9
    ExaModels.@add_con(c,  x[17] - x[18] - x[30] + T(19))  # N10
    ExaModels.@add_con(c,  x[19] - x[20] - x[31] + T(19))  # N11

    # Capacity constraints CNST(I) (L type): linear <= CIJE
    # CNST1: 20*X4,1(1) + 80*X1,4(2) <= 999.99
    # CNST2: 80*X4,1(1) + 20*X1,4(2) <= 999.99
    # CNST3: 20*X4,2(3) + 80*X2,4(4) <= 999.99
    # CNST4: 80*X4,2(3) + 20*X2,4(4) <= 999.99
    # CNST5: 20*X4,3(5) + 80*X3,4(6) <= 999.99
    # CNST6: 80*X4,3(5) + 20*X3,4(6) <= 999.99
    # CNST7: 20*X5,6(9) + 80*X6,5(10) <= 999.99
    # CNST8: 80*X5,6(9) + 20*X6,5(10) <= 999.99
    # CNST9: 20*X5,7(11) + 80*X7,5(12) <= 999.99
    # CNST10: 80*X5,7(11) + 20*X7,5(12) <= 999.99
    # CNST11: 20*X8,9(15) + 80*X9,8(16) <= 999.99
    # CNST12: 80*X8,9(15) + 20*X9,8(16) <= 999.99
    # CNST13: 20*X8,10(17) + 80*X10,8(18) <= 999.99
    # CNST14: 80*X8,10(17) + 20*X10,8(18) <= 999.99
    # CNST15: 20*X8,11(19) + 80*X11,8(20) <= 999.99
    # CNST16: 80*X8,11(19) + 20*X11,8(20) <= 999.99
    # CNST17: 20*X4,5(7) + 80*X5,4(8) <= 9999.99
    # CNST18: 80*X4,5(7) + 20*X5,4(8) <= 9999.99
    # CNST19: 20*X5,8(13) + 80*X8,5(14) <= 9999.99
    # CNST20: 80*X5,8(13) + 20*X8,5(14) <= 9999.99
    cnst_data_999 = [
        (a=1,  b=2,  c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST1
        (a=1,  b=2,  c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST2
        (a=3,  b=4,  c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST3
        (a=3,  b=4,  c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST4
        (a=5,  b=6,  c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST5
        (a=5,  b=6,  c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST6
        (a=9,  b=10, c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST7
        (a=9,  b=10, c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST8
        (a=11, b=12, c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST9
        (a=11, b=12, c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST10
        (a=15, b=16, c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST11
        (a=15, b=16, c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST12
        (a=17, b=18, c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST13
        (a=17, b=18, c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST14
        (a=19, b=20, c1=T(20), c2=T(80), rhs=T(999.99)),  # CNST15
        (a=19, b=20, c1=T(80), c2=T(20), rhs=T(999.99)),  # CNST16
        (a=7,  b=8,  c1=T(20), c2=T(80), rhs=T(9999.99)), # CNST17
        (a=7,  b=8,  c1=T(80), c2=T(20), rhs=T(9999.99)), # CNST18
        (a=13, b=14, c1=T(20), c2=T(80), rhs=T(9999.99)), # CNST19
        (a=13, b=14, c1=T(80), c2=T(20), rhs=T(9999.99)), # CNST20
    ]
    ExaModels.@add_con(c, d.c1 * x[d.a] + d.c2 * x[d.b] - d.rhs
                       for d in cnst_data_999;
                       lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
