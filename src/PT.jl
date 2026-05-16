# PT: Minimax LP with 2 variables and 501 G-type constraints (M=500)
# min u  s.t. u + XCOEFF(i)*x >= W(i)*(1-W(i)) for i=0..M
# where W(i) = i/M, XCOEFF(i) = W(1-W) - (2W^2-1) = W - 2W^2 + 1 - W + W^2 - 1 + ...
# phi(x,w) = (2w^2-1)*x + w*(1-w)*(1-x)
# Constraint: u - phi(x,w) >= 0, i.e. u + (w*(1-w) - (2w^2-1))*x - w*(1-w) >= 0
# Source: E. Polak and A. L. Tits, SIF input: Nick Gould, February 1994
function pt_model(; T = Float64, backend = nothing, kwargs...)
    M = 500
    H = T(1) / T(M)
    # Precompute coefficients for all i=0..M
    # W(i) = i*H, XCOEFF(i) = W(i)*(1-W(i)) - (2*W(i)^2-1)
    # RHS(i) = W(i)*(1-W(i))
    pt_data = [(i = i, xcoeff = (T(i)*H)*(1-T(i)*H) - (2*(T(i)*H)^2 - 1),
                rhs = (T(i)*H)*(1-T(i)*H)) for i in 0:M]
    c = ExaModels.ExaCore(T; backend = backend)
    # Variables: U (index 1), X (index 2); both free (FR DEFAULT)
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf))
    # Objective: min U = x[1]
    ExaModels.@add_obj(c, x[1])
    # 501 G-type constraints: x[1] + xcoeff*x[2] - rhs >= 0
    ExaModels.@add_par(c, ptd_xcoeff, T[d.xcoeff for d in pt_data])
    ExaModels.@add_par(c, ptd_rhs, T[d.rhs for d in pt_data])
    ExaModels.@add_con(c, x[1] + ptd_xcoeff[i]*x[2] - ptd_rhs[i] for i in 1:M+1;
                       lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
