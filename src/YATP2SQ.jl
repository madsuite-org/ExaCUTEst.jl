# YATP2SQ: Another test problem with double pseudo-stochastic constraints on a square matrix
# N=350 (default), NV=N^2+2N variables, NC=N^2+2N equality constraints
# Variables: X(I,J) for I,J=1..N, then Y(I) for I=1..N, then Z(I) for I=1..N
# E(I,J): x_ij - (y_i + z_j)*(1 + cos(x_ij)) = A
# ER(I): sum_{J=1}^N (x_ij + sin(x_ij)) = 1
# EC(J): sum_{I=1}^N (x_ij + sin(x_ij)) = 1
# A = 1.0, start: X(I,J)=10, Y(I)=Z(I)=0
# classification NOR2-AN-V-V
# Source: Ph. Toint, June 2003
function yatp2sq_model(N = 350; T = Float64, backend = nothing, kwargs...)
    A_val = T(1)
    nvar = N^2 + 2 * N

    x0 = vcat(fill(T(10), N^2), fill(T(0), 2 * N))

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar;
        lvar  = fill(T(-Inf), nvar),
        uvar  = fill(T(Inf), nvar),
        start = x0)

    # E(I,J) constraints: x_ij - (y_i + z_j)*(1 + cos(x_ij)) - A = 0
    # Variable layout: X(I,J) = x[(I-1)*N+J], Y(I) = x[N^2+I], Z(J) = x[N^2+N+J]
    # DC(I,J) element: ATP2 with IV XX=X(I,J), YPZ=Y(I)+Z(J)
    #   F = YPZ * cos(XX), so element = (Y(I)+Z(J))*cos(X(I,J))
    # Group E(I,J) = X(I,J) - Y(I) - Z(J) - DC(I,J) - A
    #   = x_ij - y_i - z_j - (y_i+z_j)*cos(x_ij) - A
    #   = x_ij - (y_i+z_j)*(1+cos(x_ij)) - A = 0
    ExaModels.@add_par(c, ij_data, [(xi = (i-1)*N+j, yi = N^2+i, zj = N^2+N+j)
                                     for i in 1:N for j in 1:N])
    ExaModels.@add_con(c,
        x[d.xi] - (x[d.yi] + x[d.zj]) * (T(1) + cos(x[d.xi])) - A_val
        for d in ij_data)

    # ER(I) constraints: sum_{J=1}^N (x_ij + sin(x_ij)) - 1 = 0
    # SX(I,J) element: SINX, so sin(X(I,J)) added to ER(I)
    # Group ER(I) = sum_{J=1}^N X(I,J) + sum_{J=1}^N sin(X(I,J)) - 1
    for i in 1:N
        xi_row = [(i-1)*N + j for j in 1:N]
        ExaModels.@add_par(c, xi_par_r, [(xi = k,) for k in xi_row])
        ExaModels.@add_con(c,
            sum(x[d.xi] + sin(x[d.xi]) for d in xi_par_r) - T(1))
    end

    # EC(J) constraints: sum_{I=1}^N (x_ij + sin(x_ij)) - 1 = 0
    for j in 1:N
        xi_col = [(i-1)*N + j for i in 1:N]
        ExaModels.@add_par(c, xi_par_c, [(xi = k,) for k in xi_col])
        ExaModels.@add_con(c,
            sum(x[d.xi] + sin(x[d.xi]) for d in xi_par_c) - T(1))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
