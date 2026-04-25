# CANTILVR: Minimum weight cantilever, 5 sections of square tube
# Source: W.K. Zhang and C. Fleury, LLN, 1994; SIF: Ph. Toint, Nov 1994
#
# min  0.0624*(X1+X2+X3+X4+X5)
# s.t. 61/X1^3 + 37/X2^3 + 19/X3^3 + 7/X4^3 + 1/X5^3 - 1 ≤ 0
#      X(i) ≥ 1e-6
#
# CONST group (L type): elements INVCUBE(Xi) with coefficients 61,37,19,7,1
# v = sum coeff/Xi^3 - 1.0, L type → lcon=-Inf, ucon=0
function cantilvr_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5;
        lvar  = fill(T(1e-6), 5),
        uvar  = fill(T(Inf), 5),
        start = fill(T(1.0), 5))
    ExaModels.@add_obj(c, T(0.0624) * x[i] for i in 1:5)
    ExaModels.@add_con(c,
        T(61) / x[1]^3 + T(37) / x[2]^3 + T(19) / x[3]^3 + T(7) / x[4]^3 + T(1) / x[5]^3 - T(1);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
