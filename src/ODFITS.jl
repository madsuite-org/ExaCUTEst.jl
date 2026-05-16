# ODFITS: Origin/Destination matrix fit problem
# 10 variables, 6 equality constraints
# Objective: entropic terms x*log(x/den) with scales
# Constraints: linear: F(i) = P(ij)*T(j) - sum
function odfits_model(; T = Float64, backend = nothing, kwargs...)
    # Data
    APV13 = T(90.0)
    APV14 = T(450.0)
    APV23 = T(360.0)
    MU13  = T(0.5)
    MU14  = T(0.5)
    MU23  = T(0.5)
    ENTROP = T(0.2)
    GAMMA  = T(1.5)

    TC = T[100.0, 500.0, 400.0, 1100.0, 600.0, 700.0]
    # P[i, arc] = path weight for arc
    # T13, T14, T23, T24 → x[1..4]; F(1..6) → x[5..10]
    # Arc flow: F(i) = P13(i)*T13 + P14(i)*T14 + P23(i)*T23 + P24(i)*T24
    P13 = T[1, 0, 0, 0, 0, 0]
    P14 = T[0, 1, 0, 1, 0, 0]
    P23 = T[0, 0, 1, 1, 1, 0]
    P24 = T[0, 0, 0, 1, 1, 1]

    # Scales (inverse of coeff)
    sc_T13 = T(1.0 / MU13)
    sc_T14 = T(1.0 / MU14)
    sc_T23 = T(1.0 / MU23)
    sc_T24 = T(1.0 / ENTROP)
    sc_F   = T[GAMMA / 1.0 for _ in 1:6]  # all QLT = 1

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # Variables: T13(1), T14(2), T23(3), T24(4), F1..F6 (5..10)
    # All variables have lvar = 0.1
    ExaModels.@add_var(c, x, 10; lvar = T(0.1), start = T[APV13, APV14, APV23, T(1.0), TC[1], TC[2], TC[3], TC[4], TC[5], TC[6]])

    # Objective: AP13 = T13*log(T13/APV13)/MU13; etc
    # Group AP13: scale=1/MU13, element XLOGX(T13, APV13), identity group type
    # contribution = x*log(x/den) / (1/sc) = sc_T13 * (x*log(x/den)) ... but wait:
    # The group function is identity, so contribution = v where v = linear + elements
    # v_AP13 = -T13 + T13*log(T13/APV13) = T13*(log(T13/APV13) - 1) → but scale divides v
    # Actually: AP13 group has linear coefficient -1 for T13, and constant rhs=0 (none set)
    # plus element XLOGX which contributes x*log(x/den) where den=APV13
    # Group type is identity (no GTYPE), so F(v) = v
    # Group SCALE = 1/MU13 = 2, so contribution = v/scale = v * MU13
    # v = -T13 + T13*log(T13/APV13) = T13*(log(T13/APV13) - 1)
    # contribution = MU13 * T13 * (log(T13/APV13) - 1)
    ExaModels.@add_obj(c, MU13 * (x[1] * log(x[1] / APV13) - x[1]))
    ExaModels.@add_obj(c, MU14 * (x[2] * log(x[2] / APV14) - x[2]))
    ExaModels.@add_obj(c, MU23 * (x[3] * log(x[3] / APV23) - x[3]))
    ExaModels.@add_obj(c, ENTROP * (x[4] * log(x[4]) - x[4]))
    # For arc flows CP(i): scale = GAMMA/QLT = GAMMA, group type identity
    # v_CP(i) = F(i) - 1.0 [linear term coeff=-1] + F(i)*log(F(i)/TC(i))
    # = F(i)*(log(F(i)/TC(i))-1)
    # contribution = (1/GAMMA) * F(i)*(log(F(i)/TC(i))-1)
    ExaModels.@add_par(c, tci, TC)
    ExaModels.@add_obj(c, (T(1)/GAMMA) * (x[4+i] * log(x[4+i] / tci[i]) - x[4+i]) for i in 1:6)

    # Equality constraints: F(i) - P13(i)*T13 - P14(i)*T14 - P23(i)*T23 - P24(i)*T24 = 0
    ExaModels.@add_par(c, p13, P13)
    ExaModels.@add_par(c, p14, P14)
    ExaModels.@add_par(c, p23, P23)
    ExaModels.@add_par(c, p24, P24)
    ExaModels.@add_con(c, x[4+i] - p13[i]*x[1] - p14[i]*x[2] - p23[i]*x[3] - p24[i]*x[4] for i in 1:6)

    return ExaModels.ExaModel(c; kwargs...)
end
