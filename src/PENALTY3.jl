# PENALTY3: Gill-Murray-Pitfield penalty problem
# N vars (N=200 default), free, start alternating (+1,-1)
# A = 0.001
# REXP: sum_I A*(U1_I-1)^2*exp(X(N))  where U1_I = X(I)+2*X(I+1)+10*X(I+2), I=1..N-2
# SEXP: sum_J A*(U2_J-3)^2*exp(X(N-1)) where U2_J = 2*X(J)+X(J+1), J=1..N-2
# RS: sum_{I,J} A*(U1_I-1)^2*(U2_J-3)^2 for I,J=1..N-2
# T^2: sum_{I,J} (X(I)^2-N)*(X(J)^2-N) for I,J=1..N
# U: sum_{I=1}^{N/2} (X(I)-1)^2
# A: constant = -0.001
function penalty3_model(N = 200; T = Float64, backend = nothing, kwargs...)
    A = T(0.001)
    RN = T(N)

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start = Vector{T}(undef, N)
    for i in 1:N
        start[i] = isodd(i) ? T(1.0) : T(-1.0)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # REXP: sum_I A*(X(I)+2*X(I+1)+10*X(I+2)-1)^2 * exp(X(N)) for I=1..N-2
    ExaModels.@add_obj(c, A * (x[i] + T(2)*x[i+1] + T(10)*x[i+2] - T(1))^2 * exp(x[N]) for i in 1:N-2)

    # SEXP: sum_J A*(2*X(J)+X(J+1)-3)^2 * exp(X(N-1)) for J=1..N-2
    ExaModels.@add_obj(c, A * (T(2)*x[j] + x[j+1] - T(3))^2 * exp(x[N-1]) for j in 1:N-2)

    # RS: sum_{I,J} A*(U1_I-1)^2*(U2_J-3)^2 for I,J=1..N-2
    for i in 1:N-2
        ExaModels.@add_obj(c, A * (x[i] + T(2)*x[i+1] + T(10)*x[i+2] - T(1))^2 * (T(2)*x[j] + x[j+1] - T(3))^2 for j in 1:N-2)
    end

    # T^2: sum_{I,J} (X(I)^2-N)*(X(J)^2-N) for I,J=1..N
    for i in 1:N
        ExaModels.@add_obj(c, (x[i]^2 - RN) * (x[j]^2 - RN) for j in 1:N)
    end

    # U: sum_{I=1}^{N/2} (X(I)-1)^2
    ExaModels.@add_obj(c, (x[i] - T(1))^2 for i in 1:N÷2)

    # A group (constant -0.001)
    ExaModels.@add_obj(c, -A)

    return ExaModels.ExaModel(c; kwargs...)
end
