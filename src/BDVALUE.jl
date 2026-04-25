# BDVALUE: Boundary value problem (More, Garbow, Hillstrom 1981, problem 28)
# NDP=5002 variables; X(1)=X(NDP)=0 fixed; NDP-2 free interior variables
# Constraint G(I) for I=2..NDP-1: -X(I-1)+2X(I)-X(I+1) + 0.5*H^2*(X(I)+I*H+1)^3 = 0
#   where H = 1/(NDP-1)
# Start: X(I) = (I-1)*H * ((I-1)*H - 1) for interior I
function bdvalue_model(NDP = 5002; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(NDP - 1)
    halfH2 = T(0.5) * H * H
    lvar = fill(T(-Inf), NDP)
    uvar = fill(T(Inf), NDP)
    lvar[1] = uvar[1] = T(0)
    lvar[NDP] = uvar[NDP] = T(0)
    start = T[i == 1 || i == NDP ? T(0) : T(i - 1) * H * (T(i - 1) * H - T(1)) for i in 1:NDP]
    # B parameter for constraint I: B_I = I*H + 1 (1-indexed: B[i-1] for I=i)
    b_data = T[T(i) * H + T(1) for i in 2:NDP-1]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NDP; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_par(c, B, b_data)
    ExaModels.@add_con(c,
        -x[i-1] + T(2)*x[i] - x[i+1] + halfH2 * (x[i] + B[i-1])^3
        for i in 2:NDP-1)
    return ExaModels.ExaModel(c; kwargs...)
end
