# OPTCNTRL: Optimal control problem (Murtagh/Saunders 1978)
# 32 variables: x(0..10), y(0..10), u(0..9); T=10
# x(0)=10 fixed, y(0)=0 fixed, y(10)=0 fixed; x(1..9) free, x(10)≥0
# y(1..9)≥-1; u(t)∈[-0.2,0.2]; start: y(1..9)=-1, rest=0
# Objective: 0.5 * sum_t x(t)²
# 20 constraints: B(t): x(t+1)-x(t)-0.2y(t)=0; C(t): y(t+1)-y(t)+0.004x(t)-0.2u(t)+0.01y(t)²=0
# Variable order: x(0),y(0),x(1),y(1),...,x(10),y(10),u(0),...,u(9)
# x(t)=x[2t+1], y(t)=x[2t+2], u(t)=x[23+t]
# Linear B(t) constraints come before nonlinear C(t) constraints (CUTEst ordering)
function optcntrl_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    lvar = T[T(10), T(0),
             T(-Inf), T(-1), T(-Inf), T(-1), T(-Inf), T(-1), T(-Inf), T(-1),
             T(-Inf), T(-1), T(-Inf), T(-1), T(-Inf), T(-1), T(-Inf), T(-1),
             T(-Inf), T(-1), T(0), T(0),
             T(-0.2), T(-0.2), T(-0.2), T(-0.2), T(-0.2),
             T(-0.2), T(-0.2), T(-0.2), T(-0.2), T(-0.2)]
    uvar = T[T(10), T(0),
             T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf),
             T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf), T(Inf),
             T(Inf), T(Inf), T(Inf), T(0),
             T(0.2), T(0.2), T(0.2), T(0.2), T(0.2),
             T(0.2), T(0.2), T(0.2), T(0.2), T(0.2)]
    start = T[T(10), T(0),
              T(0), T(-1), T(0), T(-1), T(0), T(-1), T(0), T(-1),
              T(0), T(-1), T(0), T(-1), T(0), T(-1), T(0), T(-1),
              T(0), T(-1), T(0), T(0),
              T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0), T(0)]
    ExaModels.@add_var(c, x, 32; lvar = lvar, uvar = uvar, start = start)
    ExaModels.@add_obj(c, T(0.5)*x[2t+1]^2 for t in 0:10)
    # B(t): x(t+1) - x(t) - 0.2*y(t) = 0  (linear, appears first in CUTEst)
    ExaModels.@add_con(c, x[2t+3] - x[2t+1] - T(0.2)*x[2t+2] for t in 0:9)
    # C(t): y(t+1) - y(t) + 0.004*x(t) - 0.2*u(t) + 0.01*y(t)² = 0
    ExaModels.@add_con(c, x[2t+4] - x[2t+2] + T(0.004)*x[2t+1] - T(0.2)*x[23+t] + T(0.01)*x[2t+2]^2 for t in 0:9)
    return ExaModels.ExaModel(c; kwargs...)
end
