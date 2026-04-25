# OET1: Chebyshev approximation phi(w)=w^2-x1*w-x2*exp(w) on [0,2] (Oettershagen 1982)
# 3 free variables: U, X1, X2; start = 0
# Objective: U
# 2*(M+1) constraints for M=500, I=0..M, W(I) = 2*I/M:
#   LO(I): U + W^2 - W*X1 - exp(W)*X2 ≥ 0
#   UP(I): U - W^2 + W*X1 + exp(W)*X2 ≥ 0
function oet1_model(M = 500; T = Float64, backend = nothing, kwargs...)
    H = T(2) / T(M)
    w  = T[i * H for i in 0:M]
    data = [(w = w[i+1], w2 = w[i+1]^2, ew = exp(w[i+1])) for i in 0:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 3; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, x[1])
    ExaModels.@add_con(c, x[1] + d.w2 - d.w*x[2] - d.ew*x[3]
        for d in data; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] - d.w2 + d.w*x[2] + d.ew*x[3]
        for d in data; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
