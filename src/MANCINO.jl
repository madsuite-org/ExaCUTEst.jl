# MANCINO: Mancino's function with variable dimension (default N=100 variables, unconstrained)
# Source: E. Spedicato (1975); Buckley #51; Schittkowski #391 (N=30)
# SIF: N=100 (active), ALPHA=5, BETA=14.0, GAMMA=3
# Variables: X(1)..X(N), all free (FR DEFAULT)
# Groups G(I) with L2 type (no scale): contribution = v_i^2
#   v_i = BETAN * X(I) + sum_{j≠i} MANC(X(J); i, j) - CI
#   where:
#     BETAN = BETA * N = 14.0 * 100 = 1400
#     CI = (i - N/2)^GAMMA = (i - 50)^3
#     MANC(X; i, j) = sqrt(X^2 + i/j) * (sin^5(log(sqrt(X^2+i/j))) + cos^5(log(sqrt(X^2+i/j))))
# Objective: sum_i v_i^2
#
# Implementation: register MANC as a bivariate function with analytic derivatives,
# so each inner-sum element is a shallow Node2{_manc, Var{Int}, ParameterNode{Int}}.
# The Hessian sparsity computation scales O(N²) in the number of inner-sum terms;
# use N=10 in tests (see test/runtests.jl override) to keep CI times reasonable.
#
# Analytic derivatives of MANC(x,r) = v*(S^5+C^5), v=sqrt(x^2+r), L=log(v), S=sin(L), C=cos(L):
#   f = S^5+C^5, fp = 5*(S^4*C - S*C^4), fpp = 5*((C^2-S^2)*(S^3-C^3) + 3*S^2*C^2*(S+C))
#   G = f+fp,  K = fp+fpp
#   ∂/∂x  = G*x/v
#   ∂/∂r  = G/(2v)
#   ∂²/∂x² = (G*r + K*x²)/v³
#   ∂²/∂x∂r = x*(K-G)/(2v³)
#   ∂²/∂r² = (K-G)/(4v³)
function _manc(x::Real, r::Real)
    u = x*x + r
    v = sqrt(u)
    L = log(v)
    S = sin(L); C = cos(L)
    v * (S^5 + C^5)
end

function _dmanc_dx(x::Real, r::Real)
    u = x*x + r; v = sqrt(u); L = log(v)
    S = sin(L); C = cos(L)
    S2=S*S; C2=C*C; S4=S2*S2; C4=C2*C2
    fp = 5*(S4*C - S*C4)
    G  = (S4*S + C4*C) + fp
    G * x / v
end

function _dmanc_dr(x::Real, r::Real)
    u = x*x + r; v = sqrt(u); L = log(v)
    S = sin(L); C = cos(L)
    S2=S*S; C2=C*C; S4=S2*S2; C4=C2*C2
    fp = 5*(S4*C - S*C4)
    G  = (S4*S + C4*C) + fp
    G / (2 * v)
end

function _d2manc_dx2(x::Real, r::Real)
    u = x*x + r; v = sqrt(u); L = log(v)
    S = sin(L); C = cos(L)
    S2=S*S; C2=C*C; S3=S2*S; C3=C2*C; S4=S3*S; C4=C3*C
    fp  = 5*(S4*C - S*C4)
    fpp = 5*((C2-S2)*(S3-C3) + 3*S2*C2*(S+C))
    G   = (S4*S + C4*C) + fp
    K   = fp + fpp
    (G*r + K*x*x) / (v*v*v)
end

function _d2manc_dxdr(x::Real, r::Real)
    u = x*x + r; v = sqrt(u); L = log(v)
    S = sin(L); C = cos(L)
    S2=S*S; C2=C*C; S3=S2*S; C3=C2*C; S4=S3*S; C4=C3*C
    fp  = 5*(S4*C - S*C4)
    fpp = 5*((C2-S2)*(S3-C3) + 3*S2*C2*(S+C))
    G   = (S4*S + C4*C) + fp
    K   = fp + fpp
    x * (K - G) / (2 * v*v*v)
end

function _d2manc_dr2(x::Real, r::Real)
    u = x*x + r; v = sqrt(u); L = log(v)
    S = sin(L); C = cos(L)
    S2=S*S; C2=C*C; S3=S2*S; C3=C2*C; S4=S3*S; C4=C3*C
    fp  = 5*(S4*C - S*C4)
    fpp = 5*((C2-S2)*(S3-C3) + 3*S2*C2*(S+C))
    G   = (S4*S + C4*C) + fp
    K   = fp + fpp
    (K - G) / (4 * v*v*v)
end

ExaModels.@register_bivariate(_manc, _dmanc_dx, _dmanc_dr, _d2manc_dx2, _d2manc_dxdr, _d2manc_dr2)

function mancino_model(N = 100; T = Float64, backend = nothing, kwargs...)
    ALPHA = 5
    BETA  = T(14.0)
    GAMMA = 3

    # SIF parameter computations
    RN    = T(N)
    BETAN = BETA * RN              # 1400.0

    # Compute scalar A for starting point
    # A = -BETAN / (BETAN^2 - (ALPHA+1)^2*(N-1)^2)
    RN1    = T(N - 1)
    AL1    = T(ALPHA + 1)          # 6.0
    BETAN2 = BETAN * BETAN
    F0     = AL1 * AL1 * RN1 * RN1
    F2     = BETAN2 - F0
    A      = -(BETAN / F2)

    # Constant -N/2 = -50.0
    HNALF = RN * T(-0.5)

    # Precompute CI[i] = (i - N/2)^GAMMA = (i - 50)^3
    CI = T[(T(i) + HNALF)^GAMMA for i in 1:N]

    # Compute starting point: X0[i] = A * (H[i] + CI[i])
    # H[i] = sum_{j≠i} sqrt(i/j) * (sin^5(log(sqrt(i/j))) + cos^5(log(sqrt(i/j))))
    x0 = Vector{T}(undef, N)
    for i in 1:N
        RI = T(i)
        H  = zero(T)
        for j in 1:N
            j == i && continue
            RJ    = T(j)
            SQI_J = sqrt(RI / RJ)
            LIJ   = log(SQI_J)
            SIJ   = sin(LIJ); CIJ = cos(LIJ)
            H    += SQI_J * (SIJ^ALPHA + CIJ^ALPHA)
        end
        x0[i] = A * (H + CI[i])
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # For each group i: register one parameter array Ri[j] = i/j (j=1..N),
    # then add scalar objective (one @add_obj per group):
    #   (BETAN*x[i] + sum_{j=1}^N _manc(x[j],Ri[j]) - _manc(x[i],1) - CI[i])^2
    # sum(j=1..N) includes the diagonal j=i; the diagonal term _manc(x[i], Ri[i]) = _manc(x[i], 1)
    # is subtracted explicitly to match sum_{j≠i}.
    # Each inner-sum element _manc(x[j], Ri[j]) is Node2{_manc, Var{Int}, ParameterNode{Int}}
    # (depth-1, uniform type). Compilation scales O(N²); use small N in tests.
    for i in 1:N
        ratios_i = T[T(i) / T(j) for j in 1:N]
        ExaModels.@add_par(c, Ri, ratios_i)
        CI_i = CI[i]
        ExaModels.@add_obj(
            c,
            (
                BETAN * x[i]
                + sum(_manc(x[j], Ri[j]) for j in 1:N)
                - _manc(x[i], T(1))
                - CI_i
            )^2
        )
    end

    return ExaModels.ExaModel(c; kwargs...)
end
