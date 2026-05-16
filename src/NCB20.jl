# NCB20: Banded problem with semi-bandwidth 20 (with extra coupling variables)
# Source: Ph. Toint, 1992.  SIF: October 1992.  classification OUR2-AN-V-0
#
# N + NY variables: X(1..N) and Y(1..NY), all free (XR DEFAULT)
# P=20, NY=10, COND=1e4, CL=-5.0 (= float(P)/(-4) = 20/(-4))
#
# Objective = sum_{I=1}^{N} O(I) + W   (all N-type groups)
#
# Group O(I):
#   linear part: for I=1..N-P: CL * sum_{J=1}^{P} X(I+J-1)  (coefficient CL per var)
#   + constant +2.0 (from RHS=-2.0)
#   + (10/I) * BP(X(I..I+P-1))    for I=1..N-P
#   + (1/N-P+1) * ... hmm wait:  ZE O(I) E(I) 1/I means coefficient = 10/I
#   Wait, group uses section: ZE O(I) E(I) 1/I  where 1/I = RI 10.0 (RD 1/I RI 10.0 → 1/I = 10/I)
#   + X(I)^4 (QR element, coeff 1.0 from XE)
#
# Group W:
#   elements L(1..NY) each with coefficient 1/COND = 1e-4
#   L(I): 3P element: F(X,Y,Z) = X*Y*Z + 2*Z^2 where X=X(I), Y=X(NY+I), Z=Y(I)
#   + constant +2.0 (from RHS=-2.0)
#
# Variable layout (SIF declaration order):
#   X(1), X(2), ..., X(N), Y(1), Y(2), ..., Y(NY)
# Start: X(I)=0 for all, Y(I)=1.0 for I=1..NY (XV NCB20 Y(I) 1.0)

function ncb20_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    P    = 20
    NY   = 10
    COND = T(1e4)
    invCOND = T(1) / COND
    CL   = T(-4) / T(P)   # RD CL RP -4.0 where RP=P → CL = -4.0/P = -0.2

    # Variable layout: X(1..N) then Y(1..NY)
    nvar = N + NY
    XI(i) = i           # X(i) index, i=1..N
    YI(i) = N + i       # Y(i) index, i=1..NY

    lvar  = fill(T(-Inf), nvar)
    uvar  = fill(T(Inf),  nvar)
    start = zeros(T, nvar)
    for i in 1:NY
        start[YI(i)] = T(1)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # ── Constant offset: +2 per group (N groups O(I) + 1 group W) ────────────
    ExaModels.@add_obj(c, T(2) for _ in 1:N+1)

    # ── QR element: X(I)^4 for all I=1..N (from S(I) with coeff 1.0) ─────────
    ExaModels.@add_obj(c, x[XI(i)]^4 for i in 1:N)

    # ── For I=1..N-P: linear part + BP element ───────────────────────────────
    # Linear ZN loop: DO I 1 N-P, DO J 1 P: ZN O(I) X(I+J-1) CL
    # BP element ZE loop: DO I 1 N-P: ZE O(I) E(I) 1/I  where 1/I = 10/I
    # Both loops go from I=1..N-P only
    for I in 1:(N-P)
        ci = T(10) / T(I)
        ExaModels.@add_obj(c, CL * (sum(x[XI(I + j)] for j in 0:(P-1))))
        ExaModels.@add_obj(c,
            ci * (sum(x[XI(I + k)] / (T(1) + x[XI(I + k)]^2) for k in 0:(P-1)))^2)
    end

    # ── W group: sum_{I=1}^{NY} (1/COND) * (X(I)*X(NY+I)*Y(I) + 2*Y(I)^2) ──
    # 3P element: F(X,Y,Z) = X*Y*Z + 2*Z^2  (X=X(I), Y=X(NY+I), Z=Y(I))
    w_data = [(xi = XI(i), xnyi = XI(NY + i), yi = YI(i)) for i in 1:NY]
    ExaModels.@add_obj(c,
        invCOND * (x[d.xi] * x[d.xnyi] * x[d.yi] + T(2) * x[d.yi]^2)
        for d in w_data)

    return ExaModels.ExaModel(c; kwargs...)
end
