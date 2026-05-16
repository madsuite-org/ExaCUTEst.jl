# SPARSINE: Sparse sine problem (N=5000, all free, start=0.5)
# GROUP TYPE SQR(v, P): F = 0.5*P*v^2
# OBJ(I) has type SQR with P=I, and six SINE elements:
#   v_I = sin(X(I)) + sin(X(j1)) + ... + sin(X(j5))
#   where jk(I) = mod(k*I-1, N)+1 for k in {2,3,5,7,11}
# Contribution = 0.5 * I * v_I^2
function sparsine_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    data = [(i = i,
             j2  = mod(2*i  - 1, N) + 1,
             j3  = mod(3*i  - 1, N) + 1,
             j5  = mod(5*i  - 1, N) + 1,
             j7  = mod(7*i  - 1, N) + 1,
             j11 = mod(11*i - 1, N) + 1,
             p   = T(0.5) * T(i)) for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0.5))
    ExaModels.@add_obj(c,
        let v = sin(x[d.i]) + sin(x[d.j2]) + sin(x[d.j3]) + sin(x[d.j5]) + sin(x[d.j7]) + sin(x[d.j11])
            d.p * v^2
        end
        for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
