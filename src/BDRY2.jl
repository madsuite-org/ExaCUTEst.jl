# BDRY2: 2D boundary value problem (N=499 default, (N+2)^2 variables, constrained)
# Source: Hans Mittelmann; SIF input: Nick Gould, April 2012
# classification QLR2-AN-V-V
#
# Parameters: N=499, N1=N+1=500, H=1/N1, T_param=5.0, G=0.0, A=0.0
# Variables: X(I,J) for I,J in 0..N1
#
# CUTEst variable ordering (DEFAULT+XX reordering):
#   Positions 1..n-4: non-corner variables in SIF row-major order
#   Positions n-3..n: X(0,0), X(N1,0), X(0,N1), X(N1,N1) (XX declaration order)
#
# Objective: sum_{I,J in N1/4..3N1/4} 0.5*H^2*(X(I,J)-1)^2
# Constraints P(I,J): 4*X(I,J)-X(I,J+1)-X(I,J-1)-X(I+1,J)-X(I-1,J)=0  I,J in 1..N
#             B1(I): X(I,0) - X(I,1) = 0
#             B2(I): (1-H)*X(0,I) - X(1,I) + H*5 = 0
#             B3(I): (1-H)*X(N1,I) - X(N,I) + H*5 = 0
function bdry2_model(N = 499; T = Float64, backend = nothing, kwargs...)
    N1  = N + 1           # 500
    NV  = N1 + 1          # 501
    n   = NV * NV         # 251001

    H    = T(1) / T(N1)
    HT   = H * T(5)
    HH2  = T(0.5) * H * H
    omH  = T(1) - H

    # CUTEst vidx: SIF 0-based row-major pos, corners at n-3..n
    # XX order: X(0,0)→n-3, X(N1,0)→n-2, X(0,N1)→n-1, X(N1,N1)→n
    cp0 = 0
    cp1 = N1 * NV
    cp2 = N1
    cp3 = N1 * NV + N1
    vidx(i, j) = begin
        sp = i * NV + j
        if     sp == cp0; n - 3
        elseif sp == cp1; n - 2
        elseif sp == cp2; n - 1
        elseif sp == cp3; n
        else
            nc = (sp > cp0) + (sp > cp1) + (sp > cp2) + (sp > cp3)
            sp - nc + 1
        end
    end

    lvar = zeros(T, n)
    uvar = fill(T(10), n)

    iq1 = div(N1, 4)
    iq3 = 3 * iq1
    for i in iq1:iq3, j in iq1:iq3
        uvar[vidx(i, j)] = T(3.15)
    end

    lvar[vidx(0, 0)]   = T(0);  uvar[vidx(0, 0)]   = T(0)
    lvar[vidx(N1, 0)]  = T(0);  uvar[vidx(N1, 0)]  = T(0)
    lvar[vidx(0, N1)]  = T(10); uvar[vidx(0, N1)]  = T(10)
    lvar[vidx(N1, N1)] = T(10); uvar[vidx(N1, N1)] = T(10)

    # No START POINT section → all start at 0
    x0 = zeros(T, n)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, n; lvar = lvar, uvar = uvar, start = x0)

    obj_data = [(k = vidx(i, j),) for i in iq1:iq3 for j in iq1:iq3]
    ExaModels.@add_obj(c, HH2 * (x[d.k] - T(1))^2 for d in obj_data)

    P_data = [(
        ij  = vidx(i, j),
        ijp = vidx(i, j+1),
        ijm = vidx(i, j-1),
        ipj = vidx(i+1, j),
        imj = vidx(i-1, j),
    ) for i in 1:N for j in 1:N]
    ExaModels.@add_con(c,
        T(4)*x[d.ij] - x[d.ijp] - x[d.ijm] - x[d.ipj] - x[d.imj]
        for d in P_data)

    B_data = vcat([
        [(a = vidx(i, 0),   b = vidx(i, 1),   ca = T(1),    rhs = T(0)),
         (a = vidx(0, i),   b = vidx(1, i),   ca = omH,     rhs = HT ),
         (a = vidx(N1, i),  b = vidx(N, i),   ca = omH,     rhs = HT )]
        for i in 1:N
    ]...)
    ExaModels.@add_con(c, d.ca * x[d.a] - x[d.b] + d.rhs for d in B_data)

    return ExaModels.ExaModel(c; kwargs...)
end
