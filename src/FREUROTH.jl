# FREUROTH: Freudenstein and Roth problem (scalable, unconstrained, N=5000)
# minimize sum_{i=1}^{N-1} [ (x[i] - 2*x[i+1] + 5*x[i+1]^2 - x[i+1]^3 - 13)^2
#                           + (x[i] - 14*x[i+1] + x[i+1]^2 + x[i+1]^3 - 29)^2 ]
# All variables free. Start: X(1)=0.5, X(2)=-2.0, rest=0.0.
# Source: More', Garbow, Hillstrom (1981), problem 2
# SIF: Groups R(I) and S(I), L2 group type, FRDRTH element on X(I+1)
#   FRDRTH(ELV; COEFF, XCOEFF) = (COEFF + XCOEFF*ELV)*ELV^2
#   A(I): ELV=X(I+1), COEFF=5, XCOEFF=-1 → 5*x[i+1]^2 - x[i+1]^3
#   B(I): ELV=X(I+1), COEFF=1, XCOEFF=1  → x[i+1]^2 + x[i+1]^3
#   R(I): v = x[i] - 2*x[i+1] + (5*x[i+1]^2 - x[i+1]^3) - 13 → contribution = v^2
#   S(I): v = x[i] - 14*x[i+1] + (x[i+1]^2 + x[i+1]^3) - 29  → contribution = v^2
function freuroth_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    start = fill(T(0), N)
    start[1] = T(0.5)
    if N >= 2
        start[2] = T(-2.0)
    end
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = fill(T(-Inf), N), uvar = fill(T(Inf), N),
                      start = start)
    # R(I) groups: (x[i] - 2*x[i+1] + 5*x[i+1]^2 - x[i+1]^3 - 13)^2
    ExaModels.@add_obj(c, (x[i] - T(2)*x[i+1] + T(5)*x[i+1]^2 - x[i+1]^3 - T(13))^2
                       for i in 1:N-1)
    # S(I) groups: (x[i] - 14*x[i+1] + x[i+1]^2 + x[i+1]^3 - 29)^2
    ExaModels.@add_obj(c, (x[i] - T(14)*x[i+1] + x[i+1]^2 + x[i+1]^3 - T(29))^2
                       for i in 1:N-1)
    return ExaModels.ExaModel(c; kwargs...)
end
