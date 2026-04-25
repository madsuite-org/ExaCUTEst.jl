# BDRY2: 2D boundary value problem (N=499 default, (N+2)^2 variables, constrained)
# Source: Hans Mittelmann; SIF input: Nick Gould, April 2012
# classification QLR2-AN-V-V
#
# Parameters: N=499, N1=N+1=500, H=1/N1, T_param=5.0, G=0.0, A=0.0
# Variables: X(I,J) for I,J in 0..N1,  flat 1-based index: I*(N1+1)+J+1
#
# Objective (N group, identity, no group type):
#   sum_{I=N1/4}^{3N1/4} sum_{J=N1/4}^{3N1/4} 0.5*H^2 * (X(I,J) - 1)^2
#   (SQD element: F=(Y-YP)^2=(X(I,J)-1)^2, coefficient .5H2; .5HA=0 so E(I,N1) terms vanish)
#
# Constraint ordering (empirically verified against CUTEst):
#   P(I,J): 4*X(I,J)-X(I,J+1)-X(I,J-1)-X(I+1,J)-X(I-1,J) = 0   I,J in 1..N  (N^2 groups)
#   For I=1..N: B1(I), B2(I), B3(I)  (interleaved triples)
#     B1(I): X(I,0) - X(I,1) = 0
#     B2(I): (1-H)*X(0,I) - X(1,I) + H*T_param = 0
#     B3(I): (1-H)*X(N1,I) - X(N,I) + H*T_param = 0
#
# Bounds: 0 <= X(I,J) <= 10  (default)
#   Inner block I,J in N1/4..3N1/4: uvar = 3.15
#   Corners: X(0,0)=0, X(N1,0)=0, X(0,N1)=10, X(N1,N1)=10 (fixed)
function bdry2_model(N = 499; T = Float64, backend = nothing, kwargs...)
    N1  = N + 1           # 500
    NV  = N1 + 1          # number of grid points per dimension = 501
    n   = NV * NV         # total variables = 251001

    # SIF real parameters
    H    = T(1) / T(N1)
    HT   = H * T(5)       # H*T_param, T_param=5.0
    H2   = H * H
    HH2  = T(0.5) * H2   # .5H2
    omH  = T(1) - H       # 1 - H

    # Flat 1-based index: X(I,J) -> I*NV + J + 1  (I,J are 0-based)
    vidx(i, j) = i * NV + j + 1

    # --- Bounds ---
    lvar = zeros(T, n)
    uvar = fill(T(10), n)

    iq1 = div(N1, 4)      # N1/4 = 125
    iq3 = 3 * iq1         # 3*N1/4 = 375

    for i in iq1:iq3, j in iq1:iq3
        uvar[vidx(i, j)] = T(3.15)
    end

    # Fixed corners
    uvar[vidx(0, 0)]   = T(0)
    uvar[vidx(N1, 0)]  = T(0)
    lvar[vidx(0, N1)]  = T(10);  uvar[vidx(0, N1)]  = T(10)
    lvar[vidx(N1, N1)] = T(10);  uvar[vidx(N1, N1)] = T(10)

    # Start point: no START POINT section -> default = lvar
    x0 = copy(lvar)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, n; lvar = lvar, uvar = uvar, start = x0)

    # --- Objective ---
    # Inner block: I,J in N1/4..3N1/4 (inclusive)
    obj_data = [(k = vidx(i, j),) for i in iq1:iq3 for j in iq1:iq3]
    ExaModels.@add_obj(c, HH2 * (x[d.k] - T(1))^2 for d in obj_data)

    # --- Constraints ---
    # (1) P(I,J): 4*x[ij]-x[ij+1]-x[ij-1]-x[i+1,j]-x[i-1,j] = 0  for I,J in 1..N
    P_data = [(
        ij  = vidx(i, j),
        ijp = vidx(i, j+1),
        ijm = vidx(i, j-1),
        ipj = vidx(i+1, j),
        imj = vidx(i-1, j),
    ) for i in 1:N for j in 1:N]
    ExaModels.@add_con(
        c,
        T(4) * x[d.ij] - x[d.ijp] - x[d.ijm] - x[d.ipj] - x[d.imj]
        for d in P_data
    )

    # (2) For I=1..N: B1(I), B2(I), B3(I) interleaved  (3N constraints)
    # Encode as a single @add_con with a combined NamedTuple:
    #   B1(I): coeff_a * x[a] - x[b] + rhs, with coeff_a=1, a=vidx(I,0), b=vidx(I,1), rhs=0
    #   B2(I): coeff_a * x[a] - x[b] + rhs, with coeff_a=(1-H), a=vidx(0,I), b=vidx(1,I), rhs=HT
    #   B3(I): coeff_a * x[a] - x[b] + rhs, with coeff_a=(1-H), a=vidx(N1,I), b=vidx(N,I), rhs=HT
    B_data = vcat([
        [(a = vidx(i, 0),   b = vidx(i, 1),   ca = T(1),    rhs = T(0)),
         (a = vidx(0, i),   b = vidx(1, i),   ca = omH,     rhs = HT ),
         (a = vidx(N1, i),  b = vidx(N, i),   ca = omH,     rhs = HT )]
        for i in 1:N
    ]...)
    ExaModels.@add_con(c, d.ca * x[d.a] - x[d.b] + d.rhs for d in B_data)

    return ExaModels.ExaModel(c; kwargs...)
end
