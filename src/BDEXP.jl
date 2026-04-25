# BDEXP: Banded exponential problem
# Source: Conn, Gould, Lescrenier, Toint (1988), Problem 56; SIF: Ph. Toint, Dec 1989
#
# N=5000 variables, lvar=0, uvar=Inf, start=1
# DEXP element: F(V1,V2,V3) = (V1+V2)*exp(-(V1+V2)*V3)  via IV U1=V1+V2, U2=-V3
# Groups G(I) for I=1..N-2, N type, identity: contribution = F(X(I),X(I+1),X(I+2))
function bdexp_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(0), uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c,
        (x[i] + x[i + 1]) * exp(-(x[i] + x[i + 1]) * x[i + 2]) for i in 1:(N - 2))
    return ExaModels.ExaModel(c; kwargs...)
end
