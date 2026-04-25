# CLNLBEAM: Clamped nonlinear beam optimal control problem
# NI=2000 intervals → 3*(NI+1)=6003 variables, 2*NI=4000 equality constraints
# Variables (blocks of NI+1): t[1..NI+1]=T(0..NI), xx[NI+2..2NI+2]=X(0..NI), u[2NI+3..3NI+3]=U(0..NI)
# ALPHA=350, H=1/NI, AH=ALPHA*H, H2=H/2, AH2=AH/2
# Objective: sum_{I=0}^{NI-1} [H/2*(U(I)^2+U(I+1)^2) + AH/2*(cos(T(I))+cos(T(I+1)))]
# EX(I): X(I+1)-X(I) - H/2*(sin(T(I))+sin(T(I+1))) = 0  for I=0..NI-1
# ET(I): T(I+1)-T(I) - H/2*(U(I)+U(I+1)) = 0             for I=0..NI-1
# Fixed: T(0)=T(NI)=X(0)=X(NI)=0; T(I) in [-1,1], X(I) in [-0.05,0.05]
function clnlbeam_model(NI = 2000; T = Float64, backend = nothing, kwargs...)
    alpha = T(350)
    rni   = T(NI)
    h     = T(1) / rni
    h2    = h / T(2)
    ah2   = alpha * h / T(2)
    nv    = 3 * (NI + 1)

    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)

    # T(I) bounds: [-1, 1], T(0)=T(NI)=0 fixed (XX)
    for i in 0:NI
        idx = i + 1
        lvar[idx] = T(-1); uvar[idx] = T(1)
    end
    lvar[1]    = T(0); uvar[1]    = T(0)   # T(0) = 0 fixed
    lvar[NI+1] = T(0); uvar[NI+1] = T(0)   # T(NI) = 0 fixed

    # X(I) bounds: [-0.05, 0.05], X(0)=X(NI)=0 fixed (XX)
    for i in 0:NI
        idx = NI + 1 + i + 1
        lvar[idx] = T(-0.05); uvar[idx] = T(0.05)
    end
    lvar[NI+2]     = T(0); uvar[NI+2]     = T(0)   # X(0) = 0 fixed
    lvar[2*NI+2]   = T(0); uvar[2*NI+2]   = T(0)   # X(NI) = 0 fixed

    # U(I): free (indices 2*(NI+1)+1 .. 3*(NI+1))

    # Start: X(I) = T(I) = 0.05*cos(I/NI), U(I) = 0
    start = zeros(T, nv)
    for i in 0:NI
        val = T(0.05) * cos(T(i) * h)
        start[i+1]         = val   # T(I)
        start[NI+1+i+1]    = val   # X(I)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Index helper: T(I)=x[I+1], X(I)=x[NI+1+I+1], U(I)=x[2*(NI+1)+I+1]
    # Objective
    ExaModels.@add_obj(
        c,
        h2 * (x[2*(NI+1)+i+1]^2 + x[2*(NI+1)+i+2]^2) +
        ah2 * (cos(x[i+1]) + cos(x[i+2]))
        for i in 0:NI-1
    )

    # EX(I) constraints: X(I+1)-X(I)-H/2*(sin(T(I+1))+sin(T(I)))=0
    ExaModels.@add_con(
        c,
        x[NI+1+i+2] - x[NI+1+i+1] - h2 * (sin(x[i+2]) + sin(x[i+1]))
        for i in 0:NI-1
    )

    # ET(I) constraints: T(I+1)-T(I)-H/2*(U(I+1)+U(I))=0
    ExaModels.@add_con(
        c,
        x[i+2] - x[i+1] - h2 * (x[2*(NI+1)+i+2] + x[2*(NI+1)+i+1])
        for i in 0:NI-1
    )

    return ExaModels.ExaModel(c; kwargs...)
end
