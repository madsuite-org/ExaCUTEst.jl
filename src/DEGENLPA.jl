# DEGENLPA: Small degenerate linear program (20 variables, 15 equality constraints)
# Source: Kotiah and Steinberg (1977)
# All variables bounded above by 1.0 (lvar=0 by default SIF, uvar=1)
# start: all 1.0
function degenlpa_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 20; lvar = T(0), uvar = T(1), start = T(1))
    # Objective (N group):
    # OBJ: 0.01*x2 + 33.333*x3 + 100*x4 + 0.01*x5 + 33.343*x6 + 100.01*x7
    #      + 33.333*x8 + 133.33*x9 + 100*x10
    ExaModels.@add_obj(c,
        T(0.01) * x[2] + T(33.333) * x[3] + T(100.0) * x[4] + T(0.01) * x[5] +
        T(33.343) * x[6] + T(100.01) * x[7] + T(33.333) * x[8] +
        T(133.33) * x[9] + T(100.0) * x[10])
    # C1: x1+2x2+2x3+2x4+x5+2x6+2x7+x8+2x9+x10 = 0.70785
    ExaModels.@add_con(c,
        x[1] + T(2) * x[2] + T(2) * x[3] + T(2) * x[4] + x[5] +
        T(2) * x[6] + T(2) * x[7] + x[8] + T(2) * x[9] + x[10] - T(0.70785))
    # C2: -x1+300x2+0.09x3+0.03x4 = 0
    ExaModels.@add_con(c, -x[1] + T(300) * x[2] + T(0.09) * x[3] + T(0.03) * x[4])
    # C3: 0.326x1 - 101x2 + 200x5 + 0.06x6 + 0.02x7 = 0
    ExaModels.@add_con(c, T(0.326) * x[1] - T(101) * x[2] + T(200) * x[5] + T(0.06) * x[6] + T(0.02) * x[7])
    # C4: 0.0066667x1 - 1.03x3 + 200x6 + 0.06x8 + 0.02x9 = 0
    ExaModels.@add_con(c, T(0.0066667) * x[1] - T(1.03) * x[3] + T(200) * x[6] + T(0.06) * x[8] + T(0.02) * x[9])
    # C5: 6.6667e-4*x1 - 1.01*x4 + 200*x7 + 0.06*x9 + 0.02*x10 = 0
    ExaModels.@add_con(c, T(6.6667e-4) * x[1] - T(1.01) * x[4] + T(200) * x[7] + T(0.06) * x[9] + T(0.02) * x[10])
    # C6: 0.978*x2 - 201*x5 + 100*x11 + 0.03*x12 + 0.01*x13 = 0
    ExaModels.@add_con(c, T(0.978) * x[2] - T(201) * x[5] + T(100) * x[11] + T(0.03) * x[12] + T(0.01) * x[13])
    # C7: 0.01*x2 + 0.489*x3 - 101.03*x6 + 100*x12 + 0.03*x14 + 0.01*x15 = 0
    ExaModels.@add_con(c, T(0.01) * x[2] + T(0.489) * x[3] - T(101.03) * x[6] + T(100) * x[12] + T(0.03) * x[14] + T(0.01) * x[15])
    # C8: 0.001*x2 + 0.489*x4 - 101.03*x7 + 100*x13 + 0.03*x15 + 0.01*x16 = 0
    ExaModels.@add_con(c, T(0.001) * x[2] + T(0.489) * x[4] - T(101.03) * x[7] + T(100) * x[13] + T(0.03) * x[15] + T(0.01) * x[16])
    # C9: 0.001*x3 + 0.01*x4 - 1.04*x9 + 100*x15 + 0.03*x18 + 0.01*x19 = 0
    ExaModels.@add_con(c, T(0.001) * x[3] + T(0.01) * x[4] - T(1.04) * x[9] + T(100) * x[15] + T(0.03) * x[18] + T(0.01) * x[19])
    # C10: 0.02*x3 - 1.06*x8 + 100*x14 + 0.03*x17 + 0.01*x19 = 0
    ExaModels.@add_con(c, T(0.02) * x[3] - T(1.06) * x[8] + T(100) * x[14] + T(0.03) * x[17] + T(0.01) * x[19])
    # C11: 0.002*x4 - 1.02*x10 + 100*x16 + 0.03*x19 + 0.01*x20 = 0
    ExaModels.@add_con(c, T(0.002) * x[4] - T(1.02) * x[10] + T(100) * x[16] + T(0.03) * x[19] + T(0.01) * x[20])
    # C12: -2.5742e-6*x11 + 0.00252*x13 - 0.61975*x16 + 1.03*x20 = 0
    ExaModels.@add_con(c, T(-2.5742e-6) * x[11] + T(0.00252) * x[13] - T(0.61975) * x[16] + T(1.03) * x[20])
    # C13: -0.00257*x11 + 0.25221*x12 - 6.2*x14 + 1.09*x17 = 0
    ExaModels.@add_con(c, T(-0.00257) * x[11] + T(0.25221) * x[12] - T(6.2) * x[14] + T(1.09) * x[17])
    # C14: 0.00629*x11 - 0.20555*x12 - 4.1106*x13 + 101.04*x15 + 505.1*x16 - 256.72*x19 = 0
    ExaModels.@add_con(c,
        T(0.00629) * x[11] - T(0.20555) * x[12] - T(4.1106) * x[13] +
        T(101.04) * x[15] + T(505.1) * x[16] - T(256.72) * x[19])
    # C15: 0.00841*x12 - 0.08406*x13 - 0.20667*x14 + 20.658*x16 + 1.07*x18 - 10.5*x19 = 0
    ExaModels.@add_con(c,
        T(0.00841) * x[12] - T(0.08406) * x[13] - T(0.20667) * x[14] +
        T(20.658) * x[16] + T(1.07) * x[18] - T(10.5) * x[19])
    return ExaModels.ExaModel(c; kwargs...)
end
