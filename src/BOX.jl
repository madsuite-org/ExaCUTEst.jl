# BOX: Quartic function with sparsity (N=10000, all free, start=0)
# A(I): SQR, v = X(I)+X(1) → (X(I)+X(1))^2
# B(I): SQR, v = X(I)+X(N) → (X(I)+X(N))^2
# C(I): SQR, v = X(I)+X(N/2) → (X(I)+X(N/2))^2
# D: identity, v = -0.5*sum_I X(I) → -0.5*sum_I X(I) (linear)
# Q(I): QUARTIC, v = X(I) → X(I)^4
function box_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, (x[i] + x[1])^2 for i in 1:N)
    ExaModels.@add_obj(c, (x[i] + x[N])^2 for i in 1:N)
    ExaModels.@add_obj(c, (x[i] + x[N÷2])^2 for i in 1:N)
    ExaModels.@add_obj(c, -T(0.5) * x[i] for i in 1:N)
    ExaModels.@add_obj(c, x[i]^4 for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
