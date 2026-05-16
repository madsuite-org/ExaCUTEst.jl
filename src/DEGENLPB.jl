# DEGENLPB: Small degenerate LP (20 variables, 15 equality constraints)
# Same constraints as DEGENLPA, but negated objective coefficients
# Source: Kotiah and Steinberg (1977)
function degenlpb_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 20; lvar = T(0), uvar = T(1), start = T(1))
    # Objective (negated compared to DEGENLPA):
    ExaModels.@add_obj(c,
        T(-0.01) * x[2] + T(-33.333) * x[3] + T(-100.0) * x[4] + T(-0.01) * x[5] +
        T(-33.343) * x[6] + T(-100.01) * x[7] + T(-33.333) * x[8] +
        T(-133.33) * x[9] + T(-100.0) * x[10])
    # C1: same as DEGENLPA
    ExaModels.@add_con(c,
        x[1] + T(2) * x[2] + T(2) * x[3] + T(2) * x[4] + x[5] +
        T(2) * x[6] + T(2) * x[7] + x[8] + T(2) * x[9] + x[10] - T(0.70785))
    ExaModels.@add_con(c, -x[1] + T(300) * x[2] + T(0.09) * x[3] + T(0.03) * x[4])
    ExaModels.@add_con(c, T(0.326) * x[1] - T(101) * x[2] + T(200) * x[5] + T(0.06) * x[6] + T(0.02) * x[7])
    ExaModels.@add_con(c, T(0.0066667) * x[1] - T(1.03) * x[3] + T(200) * x[6] + T(0.06) * x[8] + T(0.02) * x[9])
    ExaModels.@add_con(c, T(6.6667e-4) * x[1] - T(1.01) * x[4] + T(200) * x[7] + T(0.06) * x[9] + T(0.02) * x[10])
    ExaModels.@add_con(c, T(0.978) * x[2] - T(201) * x[5] + T(100) * x[11] + T(0.03) * x[12] + T(0.01) * x[13])
    ExaModels.@add_con(c, T(0.01) * x[2] + T(0.489) * x[3] - T(101.03) * x[6] + T(100) * x[12] + T(0.03) * x[14] + T(0.01) * x[15])
    ExaModels.@add_con(c, T(0.001) * x[2] + T(0.489) * x[4] - T(101.03) * x[7] + T(100) * x[13] + T(0.03) * x[15] + T(0.01) * x[16])
    ExaModels.@add_con(c, T(0.001) * x[3] + T(0.01) * x[4] - T(1.04) * x[9] + T(100) * x[15] + T(0.03) * x[18] + T(0.01) * x[19])
    ExaModels.@add_con(c, T(0.02) * x[3] - T(1.06) * x[8] + T(100) * x[14] + T(0.03) * x[17] + T(0.01) * x[19])
    ExaModels.@add_con(c, T(0.002) * x[4] - T(1.02) * x[10] + T(100) * x[16] + T(0.03) * x[19] + T(0.01) * x[20])
    ExaModels.@add_con(c, T(-2.5742e-6) * x[11] + T(0.00252) * x[13] - T(0.61975) * x[16] + T(1.03) * x[20])
    ExaModels.@add_con(c, T(-0.00257) * x[11] + T(0.25221) * x[12] - T(6.2) * x[14] + T(1.09) * x[17])
    ExaModels.@add_con(c,
        T(0.00629) * x[11] - T(0.20555) * x[12] - T(4.1106) * x[13] +
        T(101.04) * x[15] + T(505.1) * x[16] - T(256.72) * x[19])
    ExaModels.@add_con(c,
        T(0.00841) * x[12] - T(0.08406) * x[13] - T(0.20667) * x[14] +
        T(20.658) * x[16] + T(1.07) * x[18] - T(10.5) * x[19])
    return ExaModels.ExaModel(c; kwargs...)
end
