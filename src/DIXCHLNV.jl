# DIXCHLNV: Scalable Dixon challenge problem with log constraints (Dixon 1991)
# N=1000 variables with lvar=1e-8; N/2 equality constraints log(x[1]*...*x[2k])=0
# Objective same structure as DIXCHLNG but summed over I=1..N-3
# Start: x[2k-1]=k+1, x[2k]=1/(k+1) for k=1..N/2
function dixchlnv_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    start = T[isodd(i) ? T(i ÷ 2 + 1) : T(1) / T(i ÷ 2 + 1) for i in 1:N]
    # Constraint data: for P(2k), accumulate log(x[j]) for j=1..2k
    # con_idx[k] = k (constraint index for P(2k))
    data_con = [(i = k, j = j) for k in 1:N÷2, j in 1:2k][:]
    data_I = [(i = i, ip1 = i + 1, ip2 = i + 2, ip3 = i + 3) for i in 1:N-3]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(1e-8), uvar = T(Inf), start = start)
    ExaModels.@add_obj(c,
        T(100) * (x[d.ip1] - x[d.i]^2)^2 + (x[d.i] - T(1))^2 +
        T(90) * (x[d.ip3] - x[d.ip2]^2)^2 + (x[d.ip2] - T(1))^2 +
        T(10.1) * (x[d.ip1] - T(1))^2 + T(10.1) * (x[d.ip3] - T(1))^2 +
        T(19.8) * ((x[d.ip1] - T(1)) * (x[d.ip3] - T(1)))^2
        for d in data_I)
    ExaModels.@add_con(c, g, N÷2)
    ExaModels.@add_con!(c, g[d.i] += log(x[d.j]) for d in data_con)
    return ExaModels.ExaModel(c; kwargs...)
end
