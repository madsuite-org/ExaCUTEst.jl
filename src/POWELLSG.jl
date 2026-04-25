# POWELLSG: Extended Powell singular problem (N variables, N multiple of 4)
# Objective: sum over k=1..N/4 of 4 terms (i = 4k-3):
#   G(i):   L2, v = x[i] + 10*x[i+1],      no scale → v^2
#   G(i+1): L2, v = x[i+2] - x[i+3], scale=0.2 → v^2/0.2 = 5*(x[i+2]-x[i+3])^2
#   G(i+2): L4, v = x[i+1] - 2*x[i+2],     no scale → v^4
#   G(i+3): L4, v = x[i] - x[i+3],    scale=0.1 → v^4/0.1 = 10*(x[i]-x[i+3])^4
# All variables are free. Default N = 5000.
# Source: More', Garbow, Hillstrom (1981), problem 13
function powellsg_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    @assert mod(N, 4) == 0 "N must be a multiple of 4"
    M = N ÷ 4  # number of 4-group blocks

    # Starting point: x[4k-3]=3, x[4k-2]=-1, x[4k-1]=0, x[4k]=1 for k=1..M
    start = T[
        let r = mod(i - 1, 4)
            r == 0 ? 3.0 : r == 1 ? -1.0 : r == 2 ? 0.0 : 1.0
        end
        for i in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # G(i): (x[i] + 10*x[i+1])^2  for i = 1,5,9,...
    ExaModels.@add_obj(c, (x[4k-3] + 10 * x[4k-2])^2 for k in 1:M)

    # G(i+1): 5*(x[i+2] - x[i+3])^2  for i = 1,5,9,...
    ExaModels.@add_obj(c, T(5) * (x[4k-1] - x[4k])^2 for k in 1:M)

    # G(i+2): (x[i+1] - 2*x[i+2])^4  for i = 1,5,9,...
    ExaModels.@add_obj(c, (x[4k-2] - 2 * x[4k-1])^4 for k in 1:M)

    # G(i+3): 10*(x[i] - x[i+3])^4  for i = 1,5,9,...
    ExaModels.@add_obj(c, T(10) * (x[4k-3] - x[4k])^4 for k in 1:M)

    return ExaModels.ExaModel(c; kwargs...)
end
