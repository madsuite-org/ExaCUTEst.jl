# BLOWEYB: Nonconvex QP on charge distribution (case B: a=0.1,b=0.4,c=0.6,d=0.9)
# N=2000, same structure as BLOWEYA but different partition indices
# NA=N/10, NB=4*N/10, NC=6*N/10, ND=9*N/10
function bloweyb_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    NA = N ÷ 10
    NB = (N ÷ 10) * 4
    NC = (N ÷ 10) * 6
    ND = (N ÷ 10) * 9

    V = Vector{T}(undef, N + 1)
    for i in 0:NA
        V[i+1] = T(1)
    end
    step1 = T(2) / T(NB - NA)
    for i in NA+1:NB
        V[i+1] = T(1) - step1 * T(i - NA)
    end
    for i in NB+1:NC
        V[i+1] = T(-1)
    end
    step2 = T(2) / T(ND - NC)
    for i in NC+1:ND
        V[i+1] = T(-1) + step2 * T(i - NC)
    end
    for i in ND+1:N
        V[i+1] = T(1)
    end

    inv_N2 = T(1) / T(N)^2

    int_rhs = T(0.5) * V[1]
    for i in 1:N-1
        int_rhs += V[i+1]
    end
    int_rhs += T(0.5) * V[N+1]

    u_idx(i) = 2 * i + 1
    w_idx(i) = 2 * i + 2
    NV = 2 * (N + 1)

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start_vals = zeros(T, NV)

    for i in 0:N
        lvar[u_idx(i)] = T(-1)
        uvar[u_idx(i)] = T(1)
        start_vals[u_idx(i)] = V[i+1]
    end

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NV; lvar = lvar, uvar = uvar, start = start_vals)

    c_u = [-V[i+1] * inv_N2 for i in 0:N]
    c_w = [-T(2) * V[i+1] * inv_N2 for i in 0:N]

    c_u[0+1] += V[1+1] - V[0+1]
    for i in 1:N-1
        c_u[i+1] += -T(2)*V[i+1] + V[i] + V[i+2]
    end
    c_u[N+1] += V[N] - V[N+1]

    lin_data_u = [(i = u_idx(i), cu = c_u[i+1]) for i in 0:N]
    ExaModels.@add_obj(ec, d.cu * x[d.i] for d in lin_data_u)

    lin_data_w = [(i = w_idx(i), cw = c_w[i+1]) for i in 0:N]
    ExaModels.@add_obj(ec, d.cw * x[d.i] for d in lin_data_w)

    ExaModels.@add_obj(ec, x[u_idx(0)]^2)
    ExaModels.@add_obj(ec, T(-2) * x[u_idx(0)] * x[u_idx(1)])

    do_data = [(ui = u_idx(i), ui1 = u_idx(i+1)) for i in 1:N-1]
    ExaModels.@add_obj(ec, T(2) * x[d.ui]^2 for d in do_data)
    ExaModels.@add_obj(ec, T(-2) * x[d.ui] * x[d.ui1] for d in do_data)

    ExaModels.@add_obj(ec, x[u_idx(N)]^2)

    c_data = [(ui = u_idx(i), wi = w_idx(i)) for i in 0:N]
    ExaModels.@add_obj(ec, inv_N2 * x[d.ui] * x[d.wi] for d in c_data)

    int_coefs = [i == 0 || i == N ? T(0.5) : T(1) for i in 0:N]

    ExaModels.@add_con(ec,
        sum(int_coefs[i+1] * x[u_idx(i)] for i in 0:N) - int_rhs)

    ExaModels.@add_con(ec, x[u_idx(0)] - x[u_idx(1)] - inv_N2 * x[w_idx(0)])

    con_data = [(ui = u_idx(i), uim1 = u_idx(i-1), uip1 = u_idx(i+1), wi = w_idx(i))
                for i in 1:N-1]
    ExaModels.@add_con(ec,
        T(2)*x[d.ui] - x[d.uip1] - x[d.uim1] - inv_N2 * x[d.wi]
        for d in con_data)

    ExaModels.@add_con(ec, x[u_idx(N)] - x[u_idx(N-1)] - inv_N2 * x[w_idx(N)])

    return ExaModels.ExaModel(ec; kwargs...)
end
