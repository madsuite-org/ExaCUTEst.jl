# MANNE: econometric equilibrium, T=2000 default, N=3T=6000 variables
# Variables: C(I), I(I), K(I) for I=1..T
# Index: x[I] = C(I), x[T+I] = I(I), x[2T+I] = K(I) for I=1..T
# Parameters computed from SIF:
#   GROW=0.03, BETA=0.95, XK0=3.0, XC0=0.95, XI0=0.05, B=0.25
#   A = (XC0+XI0)/XK0^B, GFAC = exp(log(1+GROW)*(1-B))
#   BT(I) = BETA^I / (I=T: BETA^T/(1-BETA)), AT(I) = A*GFAC^I
# Objective: sum_I BT(I)*log(C(I)) (unconstrained except bounds)
# Nonlinear constraints NL(I) G: C(I) + I(I) - AT(I)*K(I)^B >= 0, i.e.:
#   -C(I) - I(I) + AT(I)*K(I)^B <= 0 → lcon=-Inf, ucon=0
# Wait: XG NL(I) C(I) -1.0 I(I) -1.0 → v = -C-I; element KS(I) = AT(I)*K(I)^B coeff +1
# So v_g = -C(I) - I(I) + AT(I)*K(I)^B; G type: >= 0
# Linear constraints L(I): K(I+1) - K(I) - I(I) <= 0, I=1..T-1
# L(T): GROW*K(T) - I(T) <= 0
# Bounds: K(1) = 3.05 (fixed), K(I) >= 3.05 for I=2..T
#         C(I) >= 0.95, I(I) in [0.05, 1.04^I*...] (upper bound increases)
function manne_model(T_p = 2000; T = Float64, backend = nothing, kwargs...)
    GROW = T(0.03)
    BETA = T(0.95)
    XK0 = T(3.0)
    XC0 = T(0.95)
    XI0 = T(0.05)
    B = T(0.25)

    LOGXK = log(XK0)
    BLOGX = B * LOGXK
    XK0B = exp(BLOGX)
    NUM = XC0 + XI0
    A = NUM / XK0B
    ONE_MINUS_B = T(1) - B
    LOG1G = log(T(1) + GROW)
    SOME = LOG1G * ONE_MINUS_B
    GFAC = exp(SOME)

    # AT(I) = A * GFAC^I, BT(I) = BETA^I (but BT(T) *= 1/(1-BETA))
    AT = [A * GFAC^i for i in 1:T_p]
    BT = Vector{T}(undef, T_p)
    BT[1] = BETA
    for j in 2:T_p
        BT[j] = BT[j-1] * BETA
    end
    # T-th period scaled by 1/(1-BETA)
    BT[T_p] *= T(1) / (T(1) - BETA)

    # Variables: x[I]=C(I), x[T+I]=I(I), x[2T+I]=K(I) for I=1..T
    N = 3 * T_p
    lvar = Vector{T}(undef, N)
    uvar = Vector{T}(undef, N)
    start_vec = Vector{T}(undef, N)

    # C(I) >= 0.95
    for i in 1:T_p
        lvar[i] = T(0.95)
        uvar[i] = T(Inf)
        start_vec[i] = T(0.95)
    end
    # I(I) in [0.05, 1.04^I * 0.05] approximately
    iup = T(0.05)
    for i in 1:T_p
        iup *= T(1.04)
        lvar[T_p + i] = T(0.05)
        uvar[T_p + i] = iup
        start_vec[T_p + i] = T(0.05)
    end
    # K(I): K(1) fixed at 3.05
    lvar[2*T_p + 1] = T(3.05); uvar[2*T_p + 1] = T(3.05)
    start_vec[2*T_p + 1] = T(3.05)
    for i in 2:T_p
        lvar[2*T_p + i] = T(3.05)
        uvar[2*T_p + i] = T(Inf)
        start_vec[2*T_p + i] = T(3.0) + T(i - 1) * T(0.1)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start_vec)

    # Objective: sum_I BT(I)*log(C(I))
    obj_data = [(i = i, bt = T(BT[i])) for i in 1:T_p]
    ExaModels.@add_obj(c, d.bt * log(x[d.i]) for d in obj_data)

    # Nonlinear constraints NL(I): -C(I) - I(I) + AT(I)*K(I)^B >= 0
    nl_data = [(i = i, at = T(AT[i]), b = B) for i in 1:T_p]
    ExaModels.@add_con(c, -x[d.i] - x[T_p + d.i] + d.at * x[2*T_p + d.i]^d.b
                       for d in nl_data; lcon = T(0), ucon = T(Inf))

    # Linear constraints L(I) for I=1..T-1: K(I+1) - K(I) - I(I) <= 0
    lin_data = [(i = i,) for i in 1:T_p-1]
    ExaModels.@add_con(c, x[2*T_p + d.i + 1] - x[2*T_p + d.i] - x[T_p + d.i]
                       for d in lin_data; lcon = T(-Inf), ucon = T(0))

    # L(T): GROW*K(T) - I(T) <= 0
    ExaModels.@add_con(c, GROW * x[3*T_p] - x[2*T_p]; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
