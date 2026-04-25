# CURLY10: Banded function with semi-bandwidth K=10, N=10000 variables
# Unscaled version of SCURLY10: coefficient 1.0 instead of S(J)
# Source: Nick Gould, September 1997
# GROUP TYPE P4: F(v) = v*(v*(v^2 - 20) - 0.1)
# Q(I) = sum_{J=I}^{min(I+K,N)} X(J)
# Bounds: all free; start: x0[I] = 0.0001 * I / (N+1)
function curly10_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    K = 10
    RN_p1 = T(N + 1)
    x0 = T[T(0.0001) * T(i) / RN_p1 for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)
    N_K = N - K
    data_bulk = [(i = i,) for i in 1:N_K]
    ExaModels.@add_obj(
        c,
        let v = x[d.i]    + x[d.i+1]  + x[d.i+2]  + x[d.i+3]  + x[d.i+4]  +
                x[d.i+5]  + x[d.i+6]  + x[d.i+7]  + x[d.i+8]  + x[d.i+9]  +
                x[d.i+10]
            v * (v * (v^2 - T(20)) - T(0.1))
        end
        for d in data_bulk
    )
    for i in (N_K+1):N
        nterms = N - i + 1
        tail_data = [(xi = i + k - 1,) for k in 1:nterms]
        ExaModels.@add_obj(
            c,
            let v = sum(x[d2.xi] for d2 in tail_data)
                v * (v * (v^2 - T(20)) - T(0.1))
            end
        )
    end
    return ExaModels.ExaModel(c; kwargs...)
end
