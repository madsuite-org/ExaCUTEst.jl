# NCB20B: A banded problem with semi-bandwidth 20 (simplified version of NCB20)
# N=21 (original value, default here), P=20
# All variables free (XR DEFAULT), start = 0
# Constants: -2.0 (DEFAULT for all N groups → contributes +2 per group)
#
# Group O(I) value for I=1..N:
#   v_I = CL * sum_{j=0..P-1} X(I+j)  [only for I=1..N-P+1; else 0 linear part]
#         + (10/I) * (sum_{k=0..P-1} X(I+k)/(1+X(I+k)^2))^2  [only for I=1..N-P+1]
#         + 100 * X(I)^4              [for all I=1..N]
#         + 2.0                       [constant offset, from -2.0 RHS]
#
# CL = -4/P = -0.2 (for P=20)
# BP element: (sum_k V_k/(1+V_k^2))^2  (coefficient 10/I)
# QR element: XX^4                      (coefficient 100)
# Objective = sum_{I=1}^{N} v_I  (no group function, identity)
function ncb20b_model(N = 21; T = Float64, backend = nothing, kwargs...)
    P = 20
    NG = N - P + 1  # = N-19 (number of groups with BP elements)
    CL = T(-4) / T(P)  # = -0.2

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(0))

    # 1) Constant offset: +2 per group for all N groups
    # (sum of N constants, each = 2)
    # Since this is a constant, it doesn't affect the optimization but must match CUTEst.
    # ExaModels objective adds actual expressions, so we add a trivial term x[1]*0 + 2... no.
    # Actually ExaModels computes obj as sum of all @add_obj terms evaluated at x.
    # We can add a constant by just not adding it (CUTEst will include it in obj value).
    # The constant 2*N will be part of CUTEst's reported obj but ExaModels won't include it.
    # Actually looking at other implementations, constants ARE typically included.
    # We need to express +2 * N somehow. Let's use: 2*N * (x[1] - x[1] + 1) = 2*N
    # But that's hack-y. Looking at BRYBND - it has no constants section.
    # CUTEst subtracts RHS from group value: v = linear + element - RHS
    # RHS = -2.0 → v = linear + element - (-2) = linear + element + 2
    # The constant 2 is part of the group value, so it appears in the objective.
    # For CUTEst test matching, we must include it.
    # We add it as: sum over all N groups of 2, i.e., add N separate terms of 2.
    # Use a degenerate expression: 2 * x[i] / x[i] is invalid for x[i]=0.
    # Best approach: add 2*N as a scalar constant using a zero-gradient term.
    # Actually, ExaModels does support constant offsets via obj_par?
    # Looking at real code: the simplest is to just add 2.0 * x[i]^0 which is not supported.
    # Let's check: can we do T(2*N) + 0*x[1] somehow?
    # Simple approach: 2 * (x[I] - x[I] + 1) won't work.
    # Skip the constant for now - it only affects obj value but not gradient/Hessian.
    # The test checks obj at starting point (all zeros).
    # At x=0: linear=0, BP=0, QR=0, constant=2 per group → obj = 2*N
    # Without the constant, we'd get obj=0, which mismatches.
    # We must include it. Use a sum of T(2) values per group via a parameter trick.
    # Actually we can add: ExaModels.@add_obj(c, T(2*N) + T(0)*x[1]) but that's T(0)*x[1]=0.
    # Cleanest: add T(2) per group as a standalone @add_obj with no variables.
    # ExaModels @add_obj(c, expr) without iterator adds a scalar. Can we do @add_obj(c, T(2*N))?
    # Let's try with a loop: @add_obj(c, T(2) for _ in 1:N) -- this should work as N scalars.

    ExaModels.@add_obj(c, T(2) for _ in 1:N)

    # 2) QR element: 100 * X(I)^4 for all I=1..N
    ExaModels.@add_obj(c, T(100) * x[i]^4 for i in 1:N)

    # 3) For I=1..NG: linear part and BP element
    for I in 1:NG
        ci = T(10) / T(I)

        # Linear part: CL * sum_{j=0..P-1} X(I+j)
        # Add as NG separate linear sums (each a sum of P terms)
        # Use a single scalar @add_obj for each I
        ExaModels.@add_obj(c, CL * (sum(x[I + j] for j in 0:(P-1))))

        # BP element: ci * (sum_{k=0..P-1} X(I+k)/(1+X(I+k)^2))^2
        ExaModels.@add_obj(c, ci * (sum(x[I + k] / (T(1) + x[I + k]^2) for k in 0:(P-1)))^2)
    end

    return ExaModels.ExaModel(c; kwargs...)
end
