# LIPPERT2: discrete optimal flow problem, dual formulation
# NX=NY=100 default
# Variables: R (scalar, r>=0), U(I,J) for I=0..NX,J=1..NY, V(I,J) for I=1..NX,J=0..NY
# Objective: minimize R (linear)
# Conservation constraints O(I,J) E type (equality) - note: no SCALE in LIPPERT2:
#   DX*(U(I,J)-U(I-1,J)) + DY*(V(I,J)-V(I,J-1)) = S (RHS = S=1)
# Capacity constraints A/B/C/D G type (>=, so lcon=0):
#   A(I,J): -U(I,J)^2 - V(I,J)^2 + R^2 >= 0
#   B(I,J): -U(I-1,J)^2 - V(I,J)^2 + R^2 >= 0
#   C(I,J): -U(I,J)^2 - V(I,J-1)^2 + R^2 >= 0
#   D(I,J): -U(I-1,J)^2 - V(I,J-1)^2 + R^2 >= 0
# Bounds: all free, R >= 0
function lippert2_model(NX = 100, NY = 100; T = Float64, backend = nothing, kwargs...)
    DX = T(1) / T(NX)
    DY = T(1) / T(NY)
    S = T(1)

    n_U = (NX + 1) * NY
    n_V = NX * (NY + 1)
    n_total = 1 + n_U + n_V

    r_idx = 1
    u_idx(i, j) = 1 + i * NY + j   # I=0..NX, J=1..NY
    v_idx(i, j) = 1 + n_U + (i - 1) * (NY + 1) + j + 1  # I=1..NX, J=0..NY

    lvar = fill(T(-Inf), n_total)
    uvar = fill(T(Inf), n_total)
    lvar[1] = T(0)  # R >= 0
    # Start: R=1, U=varies, V=varies (from SIF ALPHA initialization)
    start_vec = fill(T(0), n_total)
    start_vec[1] = T(1)  # R=1

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, n_total; lvar = lvar, uvar = uvar, start = start_vec)

    # Objective: R
    ExaModels.@add_obj(c, x[r_idx])

    # Conservation constraints O(I,J) for I=1..NX, J=1..NY (equality = S)
    cons_data = [(i = i, j = j,
                  uij = u_idx(i, j), uim1j = u_idx(i-1, j),
                  vij = v_idx(i, j), vijm1 = v_idx(i, j-1))
                 for i in 1:NX for j in 1:NY]
    ExaModels.@add_con(c,
        DX * x[d.uij] - DX * x[d.uim1j] + DY * x[d.vij] - DY * x[d.vijm1] - S
        for d in cons_data)

    # Capacity constraints (G type): -U^2 - V^2 + R^2 >= 0
    cap_data = [(i = i, j = j,
                 pij = u_idx(i, j), pim1j = u_idx(i-1, j),
                 qij = v_idx(i, j), qijm1 = v_idx(i, j-1))
                for i in 1:NX for j in 1:NY]
    # A(I,J): -U(I,J)^2 - V(I,J)^2 + R^2 >= 0
    ExaModels.@add_con(c, -x[d.pij]^2 - x[d.qij]^2 + x[r_idx]^2
                       for d in cap_data; lcon = T(0), ucon = T(Inf))
    # B(I,J): -U(I-1,J)^2 - V(I,J)^2 + R^2 >= 0
    ExaModels.@add_con(c, -x[d.pim1j]^2 - x[d.qij]^2 + x[r_idx]^2
                       for d in cap_data; lcon = T(0), ucon = T(Inf))
    # C(I,J): -U(I,J)^2 - V(I,J-1)^2 + R^2 >= 0
    ExaModels.@add_con(c, -x[d.pij]^2 - x[d.qijm1]^2 + x[r_idx]^2
                       for d in cap_data; lcon = T(0), ucon = T(Inf))
    # D(I,J): -U(I-1,J)^2 - V(I,J-1)^2 + R^2 >= 0
    ExaModels.@add_con(c, -x[d.pim1j]^2 - x[d.qijm1]^2 + x[r_idx]^2
                       for d in cap_data; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
