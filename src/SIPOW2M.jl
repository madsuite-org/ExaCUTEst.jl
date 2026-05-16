# SIPOW2M: Semi-infinite LP discretization (M=2000 constraints, 2 variables)
# Like SIPOW1M but with different constraint set
# First M/2 constraints: cos(4*pi*(j+0.5)/M)*x1 + sin(4*pi*(j+0.5)/M)*x2 >= -1
# Last M/2 constraints: x1 >= -1 (linear)
function sipow2m_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    fourpi = T(4) * T(π)
    half = M ÷ 2
    cos_vals = T[cos(fourpi * (j + T(0.5)) / M) for j in 1:half]
    sin_vals = T[sin(fourpi * (j + T(0.5)) / M) for j in 1:half]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[0.8, 0.5])
    ExaModels.@add_obj(c, x[2])
    ExaModels.@add_par(c, cv, cos_vals)
    ExaModels.@add_par(c, sv, sin_vals)
    # First M/2 constraints: cv[j]*x1 + sv[j]*x2 + 1 >= 0
    ExaModels.@add_con(c, cv[j] * x[1] + sv[j] * x[2] + T(1) for j in 1:half;
                       lcon = T(0), ucon = T(Inf))
    # Last M/2 constraints: x1 + 1 >= 0  (x1 >= -1)
    ExaModels.@add_con(c, x[1] + T(1) for j in (half + 1):M;
                       lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
