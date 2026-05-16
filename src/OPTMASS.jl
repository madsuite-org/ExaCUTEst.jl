# OPTMASS: Constrained optimal control - particle on frictionless plane
# Variables: X(J,I), V(J,I), F(J,I) for J=1,2 and I=0..N; also X(J,N+1), V(J,N+1)
# Variable layout (matching SIF DO loops):
#   DO I=0..N, DO J=1..2: X(J,I), V(J,I), F(J,I)  (6 per step, N+1 steps)
#   DO J=1..2: X(J,N+1), V(J,N+1)  (4 final state vars)
# Total: 6*(N+1) + 4 = 6N + 10
# But N=500 default; 6*(501)+4 = 3010
function optmass_model(N = 500; T = Float64, backend = nothing, kwargs...)
    SPEED  = T(0.01)
    PEN    = T(0.335)
    invN   = T(1.0 / N)
    neg_invN = -invN
    inv2N2 = T(-0.5 / N^2)

    # Variable layout per step I=0..N: X(1,I), X(2,I), V(1,I), V(2,I), F(1,I), F(2,I)
    # Final: X(1,N+1), X(2,N+1), V(1,N+1), V(2,N+1)
    # Index: step I in 0..N → base 6*I, then +0 for X1, +1 for X2, +2 for V1, +3 for V2, +4 for F1, +5 for F2
    nfinal = 4
    nsteps = N + 1
    nvar = 6 * nsteps + nfinal

    idx_X(J, I) = 6 * I + J        # J=1,2 → 1-indexed
    idx_V(J, I) = 6 * I + 2 + J
    idx_F(J, I) = 6 * I + 4 + J
    # Final states: X(1,N+1) = 6*(N+1)+1, X(2,N+1) = 6*(N+1)+2
    #               V(1,N+1) = 6*(N+1)+3, V(2,N+1) = 6*(N+1)+4
    idx_Xf(J) = 6 * (N + 1) + J
    idx_Vf(J) = 6 * (N + 1) + 2 + J

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = fill(T(0.0), nvar)

    # Fixed: X(1,0)=0, X(2,0)=0, V(1,0)=SPEED, V(2,0)=0
    for J in 1:2
        lvar[idx_X(J, 0)] = T(0.0); uvar[idx_X(J, 0)] = T(0.0)
        lvar[idx_V(J, 0)] = J == 1 ? SPEED : T(0.0)
        uvar[idx_V(J, 0)] = J == 1 ? SPEED : T(0.0)
    end
    start[idx_V(1, 0)] = SPEED

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: -X(1,N+1)^2 - X(2,N+1)^2 + PEN*V(1,N+1)^2 + PEN*V(2,N+1)^2
    ExaModels.@add_obj(c, -x[idx_Xf(1)]^2 - x[idx_Xf(2)]^2 +
                       PEN * x[idx_Vf(1)]^2 + PEN * x[idx_Vf(2)]^2)

    # State equations A(J,I): X(J,I) - X(J,I-1) - (1/N)*V(J,I-1) - (1/2N^2)*F(J,I-1) = 0
    # A(J, N+1): involves final X,V and step N F
    for J in 1:2
        ExaModels.@add_con(c, x[idx_X(J, i)] - x[idx_X(J, i-1)] + neg_invN * x[idx_V(J, i-1)] +
                           inv2N2 * x[idx_F(J, i-1)] for i in 1:N)
        ExaModels.@add_con(c, x[idx_Xf(J)] - x[idx_X(J, N)] + neg_invN * x[idx_V(J, N)] +
                           inv2N2 * x[idx_F(J, N)])
    end

    # State equations B(J,I): V(J,I) - V(J,I-1) - (1/N)*F(J,I-1) = 0
    for J in 1:2
        ExaModels.@add_con(c, x[idx_V(J, i)] - x[idx_V(J, i-1)] + neg_invN * x[idx_F(J, i-1)]
                           for i in 1:N)
        ExaModels.@add_con(c, x[idx_Vf(J)] - x[idx_V(J, N)] + neg_invN * x[idx_F(J, N)])
    end

    # Force constraint: F(1,I)^2 + F(2,I)^2 <= 1 for I=0..N
    # SIF: XE C(I) = D(1,I) + D(2,I) with constant 1.0, so constraint = F1^2 + F2^2 - 1 = 0 (E type)
    ExaModels.@add_con(c, x[idx_F(1, i)]^2 + x[idx_F(2, i)]^2 - T(1.0) for i in 0:N)

    return ExaModels.ExaModel(c; kwargs...)
end
