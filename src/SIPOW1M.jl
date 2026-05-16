# SIPOW1M: Semi-infinite LP discretization (M=2000 constraints, 2 variables)
# Minimize x2 subject to: cos(2*pi*(j+0.5)/M)*x1 + sin(2*pi*(j+0.5)/M)*x2 >= -1
# for j=1..M
# Variables free, start x1=0.8, x2=0.5
function sipow1m_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    twopi = T(2) * T(π)
    cos_vals = T[cos(twopi * (j + T(0.5)) / M) for j in 1:M]
    sin_vals = T[sin(twopi * (j + T(0.5)) / M) for j in 1:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[0.8, 0.5])
    ExaModels.@add_obj(c, x[2])
    ExaModels.@add_par(c, cv, cos_vals)
    ExaModels.@add_par(c, sv, sin_vals)
    # C(j): cv[j]*x1 + sv[j]*x2 + 1 >= 0  (rhs=-1, G type)
    ExaModels.@add_con(c, cv[j] * x[1] + sv[j] * x[2] + T(1) for j in 1:M;
                       lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
