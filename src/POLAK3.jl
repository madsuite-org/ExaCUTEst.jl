# POLAK3: Nonlinear minmax in eleven variables (Polak, Mayne & Higgins 1991)
# 12 free variables: X(1..11), U; start = 1.0
# Objective: U; 10 L-type constraints F(I), I=1..10
# F(I): -U + sum_{J=1..11} (1/J) * exp((X(J) - sin((I-1)+2*J))²) ≤ 0
function polak3_model(; T = Float64, backend = nothing, kwargs...)
    sinvals = [sin(T(i - 1) + T(2) * T(j)) for i in 1:10, j in 1:11]
    invj = T[T(1) / T(j) for j in 1:11]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 12; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c, x[12])

    for ci in 1:10
        sv = sinvals[ci, :]
        data_j = [(sval=T(sv[j]), inv=T(invj[j]), xidx=Int32(j)) for j in 1:11]
        ExaModels.@add_con(c,
            -x[12] + sum(exp((x[d.xidx] - d.sval)^2) * d.inv for d in data_j);
            lcon = T(-Inf), ucon = T(0))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
