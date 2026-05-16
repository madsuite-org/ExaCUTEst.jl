# CONT6-QQ: Quadratic-quadratic elliptic control problem
# n=100 (default), (n+1)^2 u-vars + (n-1)^2 f-vars
# Objective: h^2 * sum_{i,j=1..n1} f[i,j]*(f[i,j] - 0.8*u[i,j])
# Constraints: PDE (nonlinear) + 4 types of boundary conditions (linear)
function cont6_qq_model(n = 100; T = Float64, backend = nothing, kwargs...)
    n1 = n - 1
    ub = T(6.09)
    r  = T(1.4)
    d  = T(1.6)
    h  = T(1) / T(n)
    h2 = h * h
    two_pi = T(2) * T(pi)
    two_pi_h2 = two_pi * h2
    SK = T(0.8)
    SM = T(1.0)
    B  = T(1.0)

    # Variable indexing:
    # u(i,j) for j=0..n, i=0..n: u_idx(i,j) = j*(n+1) + (i+1)
    # f(i,j) for j=1..n1, i=1..n1: f_idx(i,j) = NU + (j-1)*(n1) + (i-1+1)
    NU = (n + 1) * (n + 1)
    u_idx(i, j) = j * (n + 1) + (i + 1)
    f_idx(i, j) = NU + (j - 1) * n1 + i
    NF = n1 * n1
    NV = NU + NF

    lvar = fill(T(0), NV)  # default lvar=0 for u, overridden for f
    uvar = fill(T(Inf), NV)
    start_vals = fill(T(6.0), NV)  # u start = 6.0

    # u bounds: 0 <= u <= ub
    for j in 0:n, i in 0:n
        uvar[u_idx(i, j)] = ub
    end
    # f bounds: r <= f <= d, start = 1.5
    for j in 1:n1, i in 1:n1
        idx = f_idx(i, j)
        lvar[idx] = r; uvar[idx] = d
        start_vals[idx] = T(1.5)
    end

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NV; lvar = lvar, uvar = uvar, start = start_vals)

    # Objective: h2 * sum_{i,j=1..n1} f[i,j]*(SM*f[i,j] - SK*u[i,j])
    obj_data = [(fi=f_idx(i,j), ui=u_idx(i,j)) for j in 1:n1 for i in 1:n1]
    ExaModels.@add_obj(ec,
        h2 * x[d.fi] * (SM * x[d.fi] - SK * x[d.ui])
        for d in obj_data)

    # PDE constraints p(i,j) for j=1..n1, i=1..n1:
    # 4*u(i,j) - u(i+1,j) - u(i-1,j) - u(i,j+1) - u(i,j-1) - h2*u(i,j)*(A(i,j)-f(i,j)-B*u(i,j)) = 0
    # A(i,j) = 7 + 4*sin(2*pi*h2*i*j)
    pde_data = [(uij   = u_idx(i,j),
                 uip1j = u_idx(i+1,j),
                 uim1j = u_idx(i-1,j),
                 uijp1 = u_idx(i,j+1),
                 uijm1 = u_idx(i,j-1),
                 fij   = f_idx(i,j),
                 Aij   = T(7) + T(4)*sin(two_pi_h2 * T(i) * T(j)))
               for j in 1:n1 for i in 1:n1]

    ExaModels.@add_con(ec,
        T(4)*x[d.uij] - x[d.uip1j] - x[d.uim1j] - x[d.uijp1] - x[d.uijm1] -
        h2 * x[d.uij] * (d.Aij - x[d.fij] - B * x[d.uij])
        for d in pde_data)

    # bc1(i) for i=1..n1: u(i,0) - u(i,1) + h*u(i,1) = 0
    # → u(i,0) + (h-1)*u(i,1) = 0
    bc1_data = [(ui0=u_idx(i,0), ui1=u_idx(i,1)) for i in 1:n1]
    ExaModels.@add_con(ec, x[d.ui0] + (h - T(1)) * x[d.ui1] for d in bc1_data)

    # bc2(j) for j=1..n1: u(0,j) - u(1,j) + h*u(1,j) = 0
    # Note: SIF has typo "u(0,i)" - assuming it should be u(0,j)
    # → u(0,j) + (h-1)*u(1,j) = 0
    bc2_data = [(u0j=u_idx(0,j), u1j=u_idx(1,j)) for j in 1:n1]
    ExaModels.@add_con(ec, x[d.u0j] + (h - T(1)) * x[d.u1j] for d in bc2_data)

    # bc3(i) for i=1..n1: u(i,n) - u(i,n1) = 0
    bc3_data = [(uin=u_idx(i,n), uin1=u_idx(i,n1)) for i in 1:n1]
    ExaModels.@add_con(ec, x[d.uin] - x[d.uin1] for d in bc3_data)

    # bc4(j) for j=1..n1: u(n,j) - u(n1,j) = 0
    bc4_data = [(unj=u_idx(n,j), un1j=u_idx(n1,j)) for j in 1:n1]
    ExaModels.@add_con(ec, x[d.unj] - x[d.un1j] for d in bc4_data)

    return ExaModels.ExaModel(ec; kwargs...)
end
