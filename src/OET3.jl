# OET3: Chebyshev approximation phi(w)=sin(w)-(x1+x2*w+x3*w^2) on [0,1] (Oettershagen 1982)
# 4 free variables: U, X1, X2, X3; start = 0
# Objective: U
# 2*(M+1) constraints for M=500, I=0..M, W(I) = I/M:
#   LO(I): U - X1 - W*X2 - W^2*X3 + sin(W) ≥ 0
#   UP(I): U + X1 + W*X2 + W^2*X3 - sin(W) ≥ 0
function oet3_model(M = 500; T = Float64, backend = nothing, kwargs...)
    H = T(1) / T(M)
    w    = T[i * H for i in 0:M]
    data = [(w = w[i+1], w2 = w[i+1]^2, sw = sin(w[i+1])) for i in 0:M]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, x[1])
    ExaModels.@add_con(c, x[1] - x[2] - d.w*x[3] - d.w2*x[4] + d.sw
        for d in data; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + x[2] + d.w*x[3] + d.w2*x[4] - d.sw
        for d in data; lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
