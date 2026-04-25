# LOTSCHD: Economic lot scheduling quadratic program (Watson/Kuan 1993)
# 12 variables: T(I), U(I) for I=1..6, all ≥ 0; start = 0
# Objective: sum_I (X(I)*T(I))² where X=[1.502,1.126,0.815,1.268,1.502,0.740]
# 7 equality constraints (purely linear)
# Variable order: T1,U1,T2,U2,T3,U3,T4,U4,T5,U5,T6,U6
function lotschd_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 12; lvar = T(0), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, (T(1.502)*x[1])^2)
    ExaModels.@add_obj(c, (T(1.126)*x[3])^2)
    ExaModels.@add_obj(c, (T(0.815)*x[5])^2)
    ExaModels.@add_obj(c, (T(1.268)*x[7])^2)
    ExaModels.@add_obj(c, (T(1.502)*x[9])^2)
    ExaModels.@add_obj(c, (T(0.740)*x[11])^2)
    # CONS7: T1+U1+T2+U2+T3+U3+T4+U4+T5+U5+T6+U6 = 126.1
    ExaModels.@add_con(c, x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12] - T(126.1))
    # CONS1: 1.8*T1 - U1 = 11
    ExaModels.@add_con(c, T(1.8)*x[1] - x[2] - T(11.0))
    # CONS2: 2.2*T2 - 2*U2 - T3 - U3 = 3
    ExaModels.@add_con(c, T(2.2)*x[3] - T(2)*x[4] - x[5] - x[6] - T(3.0))
    # CONS3: 5.1*T3 - 2*U3 - (T1+U1+T2+U2+T4+U4+T5+U5+T6+U6) = 20
    ExaModels.@add_con(c, T(5.1)*x[5] - T(2)*x[6] - x[1] - x[2] - x[3] - x[4] - x[7] - x[8] - x[9] - x[10] - x[11] - x[12] - T(20.0))
    # CONS4: 2.2*T4 - 2*U4 - (T1+U1+T5+U5+T6+U6) = 17
    ExaModels.@add_con(c, T(2.2)*x[7] - T(2)*x[8] - x[1] - x[2] - x[9] - x[10] - x[11] - x[12] - T(17.0))
    # CONS5: 1.8*T5 - U5 - T1 - U1 = 9
    ExaModels.@add_con(c, T(1.8)*x[9] - x[10] - x[1] - x[2] - T(9.0))
    # CONS6: 7.4*T6 - U6 - (T1+U1+T2+U2+T3+U3+T4+U4+T5+U5) = 20
    ExaModels.@add_con(c, T(7.4)*x[11] - x[12] - x[1] - x[2] - x[3] - x[4] - x[5] - x[6] - x[7] - x[8] - x[9] - x[10] - T(20.0))
    return ExaModels.ExaModel(c; kwargs...)
end
