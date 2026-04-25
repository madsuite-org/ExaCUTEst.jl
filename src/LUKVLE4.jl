# LUKVLE4: Chained Cragg and Levy problem with tridiagonal constraints
# Source: Problem 5.4, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/2-1=4999 equality constraints
# All variables free; start: X(4k+1)=1, X(4k+2..4k+4)=2
#
# CUTEst variable ordering: j(m) = N+1-m except j(N/2)=N/2, j(N/2+1)=N/2+1
# j is self-inverse. Exception indices (where formula N+1-m is wrong):
#   2I+2=half at I=qtr-1, 2I=half/2I+1=half+1 at I=qtr, 2I-1=half+1 at I=qtr+1
#   I+2=half at I=half-2, I+1=half/I+2=half+1 at I=half-1
#
# Objective: A(I)=(-X(2I)+exp(X(2I-1)))^4, B(I)=100*(X(2I)-X(2I+1))^6,
#   C(I)=(u+tan(u)+6X(I+1)+8(X(I+1)^3-X(I+1)X(I))-4X(I+2)^2-2)^4 with u=X(2I+1)-X(2I+2),
#   D(I)=X(2I-1)^8, F(I)=(X(2I+2)-1)^2, using CUTEst positions j(.)
# Constraints C(K), K=N/2..N-2 (ci=1..N/2-1):
#   6X(K+1)+8(X(K+1)^3-X(K+1)X(K))-4X(K+2)^2-2 = 0, using CUTEst positions j(.)
function lukvle4_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 0 "N must be divisible by 4"
    @assert N >= 16 "N must be >= 16"
    N2m1 = N ÷ 2 - 1
    half = N ÷ 2
    qtr  = N ÷ 4

    # start[c] = sif_start[j(c)]; j is self-inverse, so j(c)=c for c=half,half+1 else N+1-c
    start = T[
        let m = (c == half || c == half + 1) ? c : N + 1 - c
            ((m - 1) % 4 == 0) ? T(1) : T(2)
        end
        for c in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Exception boundary indices
    Imex1 = qtr - 1   # 2I+2 = half  → F and C-tang wrong
    Imex2 = qtr        # 2I = half, 2I+1 = half+1  → A, B, C-tang wrong
    Imex3 = qtr + 1    # 2I-1 = half+1  → A, D wrong
    Itri1 = half - 2   # I+2 = half  → C-tridiag wrong
    Itri2 = half - 1   # I+1 = half, I+2 = half+1  → C-tridiag wrong (= N2m1)

    # A(I): v = -x[j(2I)] + exp(x[j(2I-1)]), contrib = v^4
    # clean: j(2I)=N+1-2i, j(2I-1)=N+2-2i
    ExaModels.@add_obj(c, (-x[N+1-2i] + exp(x[N+2-2i]))^4 for i in 1:(Imex2-1))
    ExaModels.@add_obj(c, (-x[half]   + exp(x[half+2]))^4)      # I=Imex2: j(half)=half,  j(half-1)=half+2
    ExaModels.@add_obj(c, (-x[half-1] + exp(x[half+1]))^4)      # I=Imex3: j(half+2)=half-1, j(half+1)=half+1
    ExaModels.@add_obj(c, (-x[N+1-2i] + exp(x[N+2-2i]))^4 for i in (Imex3+1):N2m1)

    # B(I): v = x[j(2I)] - x[j(2I+1)], contrib = 100*v^6
    # clean: j(2I)=N+1-2i, j(2I+1)=N-2i
    ExaModels.@add_obj(c, T(100)*(x[N+1-2i] - x[N-2i])^6 for i in 1:(Imex2-1))
    ExaModels.@add_obj(c, T(100)*(x[half] - x[half+1])^6)       # I=Imex2: j(half)=half, j(half+1)=half+1
    ExaModels.@add_obj(c, T(100)*(x[N+1-2i] - x[N-2i])^6 for i in (Imex2+1):N2m1)

    # C(I): u=x[j(2I+1)]-x[j(2I+2)], v=u+tan(u)+tridiag, contrib=v^4
    # clean: j(2I+1)=N-2i, j(2I+2)=N-1-2i, j(I)=N+1-i, j(I+1)=N-i, j(I+2)=N-1-i
    ExaModels.@add_obj(
        c,
        let u = x[N-2i] - x[N-1-2i]
            (u + tan(u) + T(6)*x[N-i] + T(8)*(x[N-i]^3 - x[N-i]*x[N+1-i]) -
             T(4)*x[N-1-i]^2 - T(2))^4
        end for i in 1:(Imex1-1)
    )
    # I=Imex1: j(2I+1)=half+2, j(2I+2)=half, j(I)=half+qtr+2, j(I+1)=half+qtr+1, j(I+2)=half+qtr
    ExaModels.@add_obj(c,
        let u = x[half+2] - x[half]
            (u + tan(u) + T(6)*x[half+qtr+1] + T(8)*(x[half+qtr+1]^3 - x[half+qtr+1]*x[half+qtr+2]) -
             T(4)*x[half+qtr]^2 - T(2))^4
        end
    )
    # I=Imex2: j(2I+1)=half+1, j(2I+2)=half-1, j(I)=half+qtr+1, j(I+1)=half+qtr, j(I+2)=half+qtr-1
    ExaModels.@add_obj(c,
        let u = x[half+1] - x[half-1]
            (u + tan(u) + T(6)*x[half+qtr] + T(8)*(x[half+qtr]^3 - x[half+qtr]*x[half+qtr+1]) -
             T(4)*x[half+qtr-1]^2 - T(2))^4
        end
    )
    ExaModels.@add_obj(
        c,
        let u = x[N-2i] - x[N-1-2i]
            (u + tan(u) + T(6)*x[N-i] + T(8)*(x[N-i]^3 - x[N-i]*x[N+1-i]) -
             T(4)*x[N-1-i]^2 - T(2))^4
        end for i in (Imex2+1):(Itri1-1)
    )
    # I=Itri1: j(2I+1)=4, j(2I+2)=3, j(I)=half+3, j(I+1)=half+2, j(I+2)=half (identity)
    ExaModels.@add_obj(c,
        let u = x[4] - x[3]
            (u + tan(u) + T(6)*x[half+2] + T(8)*(x[half+2]^3 - x[half+2]*x[half+3]) -
             T(4)*x[half]^2 - T(2))^4
        end
    )
    # I=Itri2: j(2I+1)=2, j(2I+2)=1, j(I)=half+2, j(I+1)=half (identity), j(I+2)=half+1 (identity)
    ExaModels.@add_obj(c,
        let u = x[2] - x[1]
            (u + tan(u) + T(6)*x[half] + T(8)*(x[half]^3 - x[half]*x[half+2]) -
             T(4)*x[half+1]^2 - T(2))^4
        end
    )

    # D(I): v = x[j(2I-1)], contrib = v^8
    # clean: j(2I-1)=N+2-2i
    ExaModels.@add_obj(c, x[N+2-2i]^8 for i in 1:(Imex3-1))
    ExaModels.@add_obj(c, x[half+1]^8)                          # I=Imex3: j(half+1)=half+1
    ExaModels.@add_obj(c, x[N+2-2i]^8 for i in (Imex3+1):N2m1)

    # F(I): v = x[j(2I+2)] - 1, contrib = v^2
    # clean: j(2I+2)=N-1-2i
    ExaModels.@add_obj(c, (x[N-1-2i] - T(1))^2 for i in 1:(Imex1-1))
    ExaModels.@add_obj(c, (x[half] - T(1))^2)                   # I=Imex1: j(half)=half
    ExaModels.@add_obj(c, (x[N-1-2i] - T(1))^2 for i in (Imex1+1):N2m1)

    # Constraints: C(K=half+ci-1) for ci=1..N2m1
    # value = 6*x[j(K+1)] + 8*(x[j(K+1)]^3 - x[j(K+1)]*x[j(K)]) - 4*x[j(K+2)]^2 - 2
    # ci=1: j(K)=half, j(K+1)=half+1, j(K+2)=half-1
    ExaModels.@add_con(c,
        T(6)*x[half+1] + T(8)*(x[half+1]^3 - x[half+1]*x[half]) - T(4)*x[half-1]^2 - T(2)
    )
    # ci=2: j(K)=half+1, j(K+1)=half-1, j(K+2)=half-2
    ExaModels.@add_con(c,
        T(6)*x[half-1] + T(8)*(x[half-1]^3 - x[half-1]*x[half+1]) - T(4)*x[half-2]^2 - T(2)
    )
    # ci=3..N2m1: j(K)=half+2-ci, j(K+1)=half+1-ci, j(K+2)=half-ci
    ExaModels.@add_con(c,
        T(6)*x[half+1-ci] + T(8)*(x[half+1-ci]^3 - x[half+1-ci]*x[half+2-ci]) -
        T(4)*x[half-ci]^2 - T(2)
        for ci in 3:N2m1
    )

    return ExaModels.ExaModel(c; kwargs...)
end
