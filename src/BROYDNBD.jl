# BROYDNBD: Broyden banded system of nonlinear equations (least squares)
# N=5000, KAPPA1=2, KAPPA2=5, KAPPA3=1, LB=5, UB=1
# All FREE, start=1.0; N equality constraints (NOR classification)
#
# Three regions with DIFFERENT element types (from GROUP USES section):
#   Upper left  (I=1..LB):        self=X(I)^3*KAPPA2,  nbrs=X(J)^2*(-KAPPA3) (all)
#   Main        (I=LB+1..N-UB-1): self=X(I)^2*KAPPA2,  LEFT=X(J)^3*(-KAPPA3), RIGHT=X(J)^2*(-KAPPA3)
#   Lower right (I=N-UB..N):      self=X(I)^3*KAPPA2,  nbrs=X(J)^2*(-KAPPA3) (all)
# Linear part (same for all): KAPPA1*X(I) - KAPPA3*sum(X(j) for j in nbrs)
function broydnbd_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    KAPPA1 = T(2)
    KAPPA2 = T(5)
    KAPPA3 = T(1)
    LB = 5
    UB = 1

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))

    # Upper left (I=1..LB): self=x[i]^3, all nbrs=x[j]^2
    for i in 1:LB
        lo = 1
        hi = min(N, i + UB)
        nbrs = [j for j in lo:hi if j != i]
        nbrs_data = [(j = j,) for j in nbrs]
        ExaModels.@add_con(
            c,
            KAPPA1 * x[i] + KAPPA2 * x[i]^3
            - KAPPA3 * sum(x[d.j] + x[d.j]^2 for d in nbrs_data)
        )
    end

    # Main (I=LB+1..N-UB-1): self=x[i]^2, LEFT nbrs=x[j]^3, RIGHT nbr=x[j]^2
    ExaModels.@add_con(
        c,
        KAPPA1 * x[i] + KAPPA2 * x[i]^2
        - KAPPA3 * (x[i-5] + x[i-5]^3)
        - KAPPA3 * (x[i-4] + x[i-4]^3)
        - KAPPA3 * (x[i-3] + x[i-3]^3)
        - KAPPA3 * (x[i-2] + x[i-2]^3)
        - KAPPA3 * (x[i-1] + x[i-1]^3)
        - KAPPA3 * (x[i+1] + x[i+1]^2)
        for i in (LB+1):(N-UB-1)
    )

    # Lower right (I=N-UB..N): self=x[i]^3, all nbrs=x[j]^2 (same structure as upper left)
    for i in (N-UB):N
        lo = max(1, i - LB)
        hi = N
        nbrs = [j for j in lo:hi if j != i]
        nbrs_data = [(j = j,) for j in nbrs]
        ExaModels.@add_con(
            c,
            KAPPA1 * x[i] + KAPPA2 * x[i]^3
            - KAPPA3 * sum(x[d.j] + x[d.j]^2 for d in nbrs_data)
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
