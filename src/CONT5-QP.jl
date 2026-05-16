# CONT5-QP: Linear-quadratic PDE control problem
# n=m=200 (default), (n+1)*(m+1) y-vars + m u-vars
# y(i,j) for j=0..n, i=0..m; u(i) for i=1..m
# Objective: quadratic tracking of y target + control cost
# Constraints: PDE p(i,j), bc1(i), bc2(i) (linear equalities)
#
# CUTEst variable ordering (determined empirically from sifdecoder):
# NP1 = n+1 = 201, NV = NP1^2 + n = 40601
#
# u(i) for i=1..n at positions NP1-i  (reversed: u(n)→1, u(1)→n)
#
# y(i,j) positions:
#   i=n, j=0:           NP1                     (= 201)
#   i=n, j=1:           2n+1                    (= 401)
#   i=n, j=2..n:        2n+2-j                  (= 400..202, reversed in j)
#   i=0..n-1, j=1:      NP1*(NP1-i)             (= 40401-201i; i=0→40401, i=n-1→402)
#   i=0, j=2..n:        j*NP1+1                 (j-outer, fixed to 0)
#   i=1..n-1, j=2..n:   j*NP1+(i+1)             (j-outer)
#   i=1..n-1, j=0:      NP1^2+n-i               (= 40601-i, reversed)
#   i=0, j=0:           NP1^2+n                 (= 40601, fixed to 0)
#   i=0, j=1:           NP1^2                   (= 40401, fixed to 0)
#
function cont5_qp_model(n = 200; T = Float64, backend = nothing, kwargs...)
    m = n
    t = T(1.58)
    a = T(0.001)

    # Derived parameters
    dx = T(1) / T(n)
    dt = t / T(m)
    h2 = dx * dx
    inv_dt = T(1) / dt
    neg_inv_dt = -inv_dt
    inv_h2 = T(1) / h2
    neg_inv_2h2 = T(-0.5) / h2
    adt = a * dt
    quarter_dx = T(0.25) * dx
    half_dx    = T(0.5)  * dx
    half_adt   = T(0.5)  * adt
    quarter_adt = T(0.25) * adt
    inv_2dx    = T(1) / (T(2) * dx)
    neg_4_inv_2dx = T(-4) * inv_2dx
    _3_inv_2dx = T(3) * inv_2dx
    _1_plus_3_inv_2dx = T(1) + _3_inv_2dx

    NP1 = n + 1  # = 201
    NV = NP1 * NP1 + n  # = 40601

    # Variable index functions matching CUTEst variable ordering
    # u(i) for i=1..n: reversed order
    u_idx(i) = NP1 - i   # u(n)→1, u(1)→n
    # y(i,j): piecewise formula
    y_idx(i, j) = begin
        if i == n
            if j == 0; return NP1
            elseif j == 1; return 2*n + 1
            else; return 2*n + 2 - j   # j=2..n
            end
        end
        if i == 0 && j == 0; return NP1^2 + n   # fixed
        elseif i == 0 && j == 1; return NP1^2    # fixed
        elseif i == 0; return j * NP1 + 1        # j=2..n, fixed
        elseif j == 0; return NP1^2 + n - i      # i=1..n-1
        elseif j == 1; return NP1 * (NP1 - i)    # i=1..n-1
        else; return j * NP1 + (i + 1)           # j=2..n, i=1..n-1
        end
    end

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start_vals = zeros(T, NV)

    # u(i) for i=1..n: lvar=0, uvar=1 (XL=0, XU=1)
    for i in 1:n
        lvar[u_idx(i)] = T(0)
        uvar[u_idx(i)] = T(1)
    end

    # y(0,j) for j=0..n: fixed to 0 (XX)
    for j in 0:n
        lvar[y_idx(0, j)] = T(0)
        uvar[y_idx(0, j)] = T(0)
    end

    # y(i,j) for i=1..n, j=0..n: lvar=-1, uvar=1 (XL=-1, XU=1)
    for i in 1:n
        for j in 0:n
            lvar[y_idx(i, j)] = T(-1)
            uvar[y_idx(i, j)] = T(1)
        end
    end

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NV; lvar = lvar, uvar = uvar, start = start_vals)

    # Objective: 0.25*dx*(y(m,0)-yt(0))^2 + sum_{j=1}^{n-1} 0.5*dx*(y(m,j)-yt(j))^2
    #           + 0.25*dx*(y(m,n)-yt(n))^2
    #           + sum_{i=1}^{m-1} 0.5*adt*u(i)^2 + 0.25*adt*u(m)^2
    # where yt(j) = 0.5*(1-(j*dx)^2)

    # Boundary j=0,n
    yt0 = T(0.5) * (T(1) - T(0))         # j=0: yt = 0.5
    ytn = T(0.5) * (T(1) - (T(n)*dx)^2)  # j=n: yt = 0.5*(1-1) = 0
    y0_idx = y_idx(m, 0)
    yn_idx = y_idx(m, n)
    ExaModels.@add_obj(ec, quarter_dx * (x[y0_idx] - yt0)^2)
    ExaModels.@add_obj(ec, quarter_dx * (x[yn_idx] - ytn)^2)

    # Interior j=1..n-1
    obj_y_data = [(yi = y_idx(m, j), yt = T(0.5)*(T(1)-(T(j)*dx)^2))
                  for j in 1:n-1]
    ExaModels.@add_obj(ec, half_dx * (x[d.yi] - d.yt)^2 for d in obj_y_data)

    # u terms: 0.5*adt*u(i)^2 for i=1..m-1, 0.25*adt*u(m)^2
    obj_u_data = [(ui = u_idx(i),) for i in 1:m-1]
    ExaModels.@add_obj(ec, half_adt * x[d.ui]^2 for d in obj_u_data)
    ExaModels.@add_obj(ec, quarter_adt * x[u_idx(m)]^2)

    # PDE constraints p(i,j) for j=1..n-1, i=0..m-1:
    # (y(i+1,j) - y(i,j))/dt - 0.5*(y(i,j-1)-2y(i,j)+y(i,j+1)+y(i+1,j-1)-2y(i+1,j)+y(i+1,j+1))/h2 = 0
    coef_ip1j  = inv_dt + inv_h2
    coef_ij    = neg_inv_dt + inv_h2
    coef_sides = neg_inv_2h2  # = -0.5/h2

    pde_data = [(y_ij    = y_idx(i, j),
                 y_ip1j  = y_idx(i+1, j),
                 y_ijm1  = y_idx(i, j-1),
                 y_ijp1  = y_idx(i, j+1),
                 y_ip1jm1 = y_idx(i+1, j-1),
                 y_ip1jp1 = y_idx(i+1, j+1))
                for j in 1:n-1 for i in 0:m-1]

    ExaModels.@add_con(ec,
        coef_ip1j * x[d.y_ip1j] + coef_ij * x[d.y_ij] +
        coef_sides * (x[d.y_ijm1] + x[d.y_ijp1] + x[d.y_ip1jm1] + x[d.y_ip1jp1])
        for d in pde_data)

    # bc1(i) for i=1..m: y(i,2) - 4*y(i,1) + 3*y(i,0) = 0
    bc1_data = [(yi0=y_idx(i,0), yi1=y_idx(i,1), yi2=y_idx(i,2)) for i in 1:m]
    ExaModels.@add_con(ec, x[d.yi2] - T(4)*x[d.yi1] + T(3)*x[d.yi0] for d in bc1_data)

    # bc2(i) for i=1..m: (y(i,n-2) - 4*y(i,n-1) + 3*y(i,n))/(2*dx) - u[i] + y(i,n) = 0
    bc2_data = [(yim2=y_idx(i,n-2), yim1=y_idx(i,n-1), yin=y_idx(i,n), ui=u_idx(i))
                for i in 1:m]
    ExaModels.@add_con(ec,
        inv_2dx * x[d.yim2] + neg_4_inv_2dx * x[d.yim1] +
        _1_plus_3_inv_2dx * x[d.yin] - x[d.ui]
        for d in bc2_data)

    return ExaModels.ExaModel(ec; kwargs...)
end
