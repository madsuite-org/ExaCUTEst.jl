# FCCU: Fluid catalytic cracker data reconciliation, 19 variables, 8 linear equality constraints
# Minimize sum_i W_i * (x_i - M_i)^2 / W_i = sum_i (x_i - M_i)^2 / W_i
# Wait: SQUARE group type with SCALE W: contribution = (x_i - M_i)^2 / W_i
# Variables: all flows >= 0, start = 1.0
# 8 equality constraints (linear mass balances)
function fccu_model(; T = Float64, backend = nothing, kwargs...)
    NV = 19
    # Variable order: Feed,Effluent,MF_ohd,HCN,LCO,HCO,MF_btms,Decant,Dec_recy,Off_gas,
    #                 DC4_feed,DC3_feed,DC4_btms,Lean_oil,Propane,Butane,C8spl_fd,LCN,MCN
    # Indices 1-19

    # Weights
    W  = T[0.2, 1, 1, 1/3, 1/3, 1/3, 1, 1, 1, 1, 1, 1, 1, 1, 1/3, 1/3, 1, 1/3, 1/3]
    # Measured flows
    M  = T[31, 36, 20, 3, 5, 3.5, 4.2, 0.9, 3.9, 2.2, 22.8, 6.8, 19, 8.5, 2.2, 2.5, 10.8, 6.5, 6.5]

    x0 = ones(T, NV)
    lvar = zeros(T, NV)   # default x >= 0
    uvar = fill(T(Inf), NV)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    # Objective: sum_i (x_i - M_i)^2 / W_i
    obj_data = [(i = i, mi = M[i], wi = W[i]) for i in 1:NV]
    ExaModels.@add_obj(c,
        (x[d.i] - d.mi)^2 / d.wi
        for d in obj_data)

    # F1: Feed + Dec_recy - Effluent = 0
    ExaModels.@add_con(c, x[1] + x[9] - x[2])
    # F2: Effluent - MF_ohd - HCN - LCO - HCO - MF_btms = 0
    ExaModels.@add_con(c, x[2] - x[3] - x[4] - x[5] - x[6] - x[7])
    # F3: MF_btms - Decant - Dec_recy = 0
    ExaModels.@add_con(c, x[7] - x[8] - x[9])
    # F4: MF_ohd + Lean_oil - Off_gas - DC4_feed = 0
    ExaModels.@add_con(c, x[3] + x[14] - x[10] - x[11])
    # F5: DC4_feed - DC3_feed - DC4_btms = 0
    ExaModels.@add_con(c, x[11] - x[12] - x[13])
    # F6: DC4_btms - Lean_oil - C8spl_fd = 0
    ExaModels.@add_con(c, x[13] - x[14] - x[17])
    # F7: DC3_feed - Propane - Butane = 0
    ExaModels.@add_con(c, x[12] - x[15] - x[16])
    # F8: C8spl_fd - LCN - MCN = 0
    ExaModels.@add_con(c, x[17] - x[18] - x[19])

    return ExaModels.ExaModel(c; kwargs...)
end
