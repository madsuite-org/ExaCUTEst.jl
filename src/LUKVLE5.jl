# LUKVLE5: Generalized Broyden tridiagonal function with five-diagonal constraints
# Source: Problem 5.5, Luksan & Vlcek (1999), TR 767
# Nvar=N+2 CUTEst variables (SIF X(0..N+1)), N-4 equality constraints
# Bounds: SIF X(0)=X(N+1)=0 fixed; all others free; start: X(1..N)=-1
#
# CUTEst variable ordering (perm of SIF index k → CUTEst pos):
#   perm(0)=N+1, perm(1)=N, perm(k)=k+1 for k=2..N-2, perm(N-1)=2, perm(N)=1, perm(N+1)=N+2
#
# Objective: sum_{I=1}^{N} |3X(I)-X(I+1)-X(I-1)-2X(I)^2+1|^(7/3), X(0)=X(N+1)=0
# Clean formula (I=3..N-3): perm(I-1)=I, perm(I)=I+1, perm(I+1)=I+2
# Individual: I=1,2 (left boundary), I=N-2,N-1,N (right boundary)
#
# Constraints: 6X(K+2)-X(K)+X(K+3)+8(X(K+2)^3-X(K+2)X(K+1))-4X(K+3)^2+X(K+1)^2-X(K+4)^2-2=0
# Clean formula (K=2..N-6): perm(K+j)=K+j+1
# Individual: K=1, K=N-5, K=N-4
function lukvle5_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N >= 7 "N must be >= 7"
    Nvar = N + 2

    # start: SIF X(1..N)=-1, X(0)=X(N+1)=0 (fixed at CUTEst pos N+1, N+2)
    start = fill(T(-1), Nvar)
    start[N+1] = T(0)
    start[N+2] = T(0)

    lvar = fill(T(-Inf), Nvar)
    uvar = fill(T(Inf), Nvar)
    lvar[N+1] = T(0); uvar[N+1] = T(0)   # SIF X(0)   → CUTEst N+1
    lvar[N+2] = T(0); uvar[N+2] = T(0)   # SIF X(N+1) → CUTEst N+2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, Nvar; lvar = lvar, uvar = uvar, start = start)

    # OBJECTIVE: |3X(I)-X(I+1)-X(I-1)-2X(I)^2+1|^(7/3)
    # I=1: perm(0)=N+1, perm(1)=N, perm(2)=3
    ExaModels.@add_obj(c, abs(T(3)*x[N] - x[3] - x[N+1] - T(2)*x[N]^2 + T(1))^(T(7)/T(3)))
    # I=2: perm(1)=N, perm(2)=3, perm(3)=4
    ExaModels.@add_obj(c, abs(T(3)*x[3] - x[4] - x[N] - T(2)*x[3]^2 + T(1))^(T(7)/T(3)))
    # I=3..N-3: perm(I-1)=I, perm(I)=I+1, perm(I+1)=I+2
    ExaModels.@add_obj(c,
        abs(T(3)*x[i+1] - x[i+2] - x[i] - T(2)*x[i+1]^2 + T(1))^(T(7)/T(3))
        for i in 3:(N-3)
    )
    # I=N-2: perm(N-3)=N-2, perm(N-2)=N-1, perm(N-1)=2
    ExaModels.@add_obj(c, abs(T(3)*x[N-1] - x[2] - x[N-2] - T(2)*x[N-1]^2 + T(1))^(T(7)/T(3)))
    # I=N-1: perm(N-2)=N-1, perm(N-1)=2, perm(N)=1
    ExaModels.@add_obj(c, abs(T(3)*x[2] - x[1] - x[N-1] - T(2)*x[2]^2 + T(1))^(T(7)/T(3)))
    # I=N: perm(N-1)=2, perm(N)=1, perm(N+1)=N+2
    ExaModels.@add_obj(c, abs(T(3)*x[1] - x[N+2] - x[2] - T(2)*x[1]^2 + T(1))^(T(7)/T(3)))

    # CONSTRAINTS: 6X(K+2)-X(K)+X(K+3)+8(X(K+2)^3-X(K+2)X(K+1))-4X(K+3)^2+X(K+1)^2-X(K+4)^2-2=0
    # K=1: perm(1)=N, perm(2)=3, perm(3)=4, perm(4)=5, perm(5)=6
    ExaModels.@add_con(c,
        T(6)*x[4] - x[N] + x[5] + T(8)*(x[4]^3 - x[4]*x[3]) -
        T(4)*x[5]^2 + x[3]^2 - x[6]^2 - T(2)
    )
    # K=2..N-6: perm(K+j)=K+j+1 (clean formula)
    ExaModels.@add_con(c,
        T(6)*x[k+3] - x[k+1] + x[k+4] + T(8)*(x[k+3]^3 - x[k+3]*x[k+2]) -
        T(4)*x[k+4]^2 + x[k+2]^2 - x[k+5]^2 - T(2)
        for k in 2:(N-6)
    )
    # K=N-5: perm(N-5)=N-4, perm(N-4)=N-3, perm(N-3)=N-2, perm(N-2)=N-1, perm(N-1)=2
    ExaModels.@add_con(c,
        T(6)*x[N-2] - x[N-4] + x[N-1] + T(8)*(x[N-2]^3 - x[N-2]*x[N-3]) -
        T(4)*x[N-1]^2 + x[N-3]^2 - x[2]^2 - T(2)
    )
    # K=N-4: perm(N-4)=N-3, perm(N-3)=N-2, perm(N-2)=N-1, perm(N-1)=2, perm(N)=1
    ExaModels.@add_con(c,
        T(6)*x[N-1] - x[N-3] + x[2] + T(8)*(x[N-1]^3 - x[N-1]*x[N-2]) -
        T(4)*x[2]^2 + x[N-2]^2 - x[1]^2 - T(2)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
