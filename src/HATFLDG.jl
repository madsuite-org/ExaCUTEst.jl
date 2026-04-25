# HATFLDG: OPTIMA test problem, nonlinear system of equations (Hatfield Poly 1989)
# 25 free variables; start = 1.0; no objective; 25 equality constraints
# G(1):    X1  - X13 - X1*X2     + 1 = 0   (A1 = 2PR with coeff -1)
# G(I):    X(I)- X13 + X(I)*(X(I-1)-X(I+1))+1=0 for I=2..24 (2PRI element)
# G(25):   X25 - X13 + X24*X25   + 1 = 0   (2PR element)
function hatfldg_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 25; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    ExaModels.@add_con(c, x[1] - x[13] - x[1]*x[2] + T(1))
    ExaModels.@add_con(c, x[i] - x[13] + x[i]*(x[i-1] - x[i+1]) + T(1) for i in 2:24)
    ExaModels.@add_con(c, x[25] - x[13] + x[24]*x[25] + T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
