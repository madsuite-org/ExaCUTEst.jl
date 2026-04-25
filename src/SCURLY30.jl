# SCURLY30: Scaled banded function with semi-bandwidth K=30, N=10000 variables
# Scaled version of CURLY30: S(I) = exp((I-1)/(N-1)*12)
# GROUP TYPE P4: F(v) = v*(v*(v^2 - 20) - 0.1)
# Q(I) = sum_{J=I}^{min(I+K,N)} S(J)*X(J)
function scurly30_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    K = 30
    SCAL = T(12.0)
    RN = T(N)
    RN_1 = RN - T(1)
    RN_p1 = RN + T(1)
    S = T[exp(T(I - 1) / RN_1 * SCAL) for I in 1:N]
    x0 = T[T(0.0001) * T(I) / RN_p1 * S[I] for I in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    ExaModels.@add_par(c, S_par, S)
    N_K = N - K
    data_bulk = [(i = i,) for i in 1:N_K]
    ExaModels.@add_obj(
        c,
        let v = S_par[d.i]    * x[d.i]    + S_par[d.i+1]  * x[d.i+1]  +
                S_par[d.i+2]  * x[d.i+2]  + S_par[d.i+3]  * x[d.i+3]  +
                S_par[d.i+4]  * x[d.i+4]  + S_par[d.i+5]  * x[d.i+5]  +
                S_par[d.i+6]  * x[d.i+6]  + S_par[d.i+7]  * x[d.i+7]  +
                S_par[d.i+8]  * x[d.i+8]  + S_par[d.i+9]  * x[d.i+9]  +
                S_par[d.i+10] * x[d.i+10] + S_par[d.i+11] * x[d.i+11] +
                S_par[d.i+12] * x[d.i+12] + S_par[d.i+13] * x[d.i+13] +
                S_par[d.i+14] * x[d.i+14] + S_par[d.i+15] * x[d.i+15] +
                S_par[d.i+16] * x[d.i+16] + S_par[d.i+17] * x[d.i+17] +
                S_par[d.i+18] * x[d.i+18] + S_par[d.i+19] * x[d.i+19] +
                S_par[d.i+20] * x[d.i+20] + S_par[d.i+21] * x[d.i+21] +
                S_par[d.i+22] * x[d.i+22] + S_par[d.i+23] * x[d.i+23] +
                S_par[d.i+24] * x[d.i+24] + S_par[d.i+25] * x[d.i+25] +
                S_par[d.i+26] * x[d.i+26] + S_par[d.i+27] * x[d.i+27] +
                S_par[d.i+28] * x[d.i+28] + S_par[d.i+29] * x[d.i+29] +
                S_par[d.i+30] * x[d.i+30]
            v * (v * (v^2 - T(20)) - T(0.1))
        end
        for d in data_bulk
    )
    for i in (N_K+1):N
        nterms = N - i + 1
        c_S = T.(S[i:N])
        tail_data = [(ci = c_S[k], xi = i + k - 1) for k in 1:nterms]
        ExaModels.@add_obj(
            c,
            let v = sum(d2.ci * x[d2.xi] for d2 in tail_data)
                v * (v * (v^2 - T(20)) - T(0.1))
            end
        )
    end
    return ExaModels.ExaModel(c; kwargs...)
end
