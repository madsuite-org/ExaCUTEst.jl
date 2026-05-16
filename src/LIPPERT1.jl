# LIPPERT1: discrete optimal flow problem, primal formulation
# NX=NY=100 default
# Variables: T (scalar), U(I,J) for I=0..NX,J=1..NY, V(I,J) for I=1..NX,J=0..NY
# Objective: minimize -T (i.e. maximize T)
# Conservation constraints O(I,J) E type (equality):
#   DX*(U(I,J)-U(I-1,J)) + DY*(V(I,J)-V(I,J-1)) - S*T = 0, scaled by DX
#   So: U(I,J) - U(I-1,J) + (DY/DX)*(V(I,J)-V(I,J-1)) - S*T/DX = 0
#   Wait: ZE O(I,J) 'SCALE' DX → constraint scaled by DX
#   v = DX*U(I,J) - DX*U(I-1,J) + DY*V(I,J) - DY*V(I,J-1) - S*T
#   SCALE = DX → residual = v/DX (for constraint) but for identity group type, just v=0
# Actually: group value v with SCALE means phi(v/sigma)=0 → v=0
# Capacity constraints A(I,J), B(I,J), C(I,J), D(I,J) L type (<=):
#   RHS = 1, so expr - 1 <= 0
#   A(I,J): U(I,J)^2 + V(I,J)^2 - 1 <= 0
#   B(I,J): U(I-1,J)^2 + V(I,J)^2 - 1 <= 0
#   C(I,J): U(I,J)^2 + V(I,J-1)^2 - 1 <= 0
#   D(I,J): U(I-1,J)^2 + V(I,J-1)^2 - 1 <= 0
# Bounds: all free, T >= 0.01
# S=1, DX=DY=1/NX=1/NY=0.01
function lippert1_model(NX = 100, NY = 100; T = Float64, backend = nothing, kwargs...)
    DX = T(1) / T(NX)
    DY = T(1) / T(NY)
    S = T(1)

    # Variable indexing:
    # x[1] = T (scalar)
    # U(I,J) for I=0..NX, J=1..NY: (NX+1)*NY variables
    # V(I,J) for I=1..NX, J=0..NY: NX*(NY+1) variables
    n_U = (NX + 1) * NY
    n_V = NX * (NY + 1)
    n_total = 1 + n_U + n_V

    u_idx(i, j) = 1 + i * NY + j   # I=0..NX, J=1..NY: offset by 1 for T
    v_idx(i, j) = 1 + n_U + (i - 1) * (NY + 1) + j + 1  # I=1..NX, J=0..NY

    lvar = fill(T(-Inf), n_total)
    uvar = fill(T(Inf), n_total)
    lvar[1] = T(0.01)  # T >= 0.01

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, n_total; lvar = lvar, uvar = uvar)

    # Objective: -T
    ExaModels.@add_obj(c, -x[1])

    # Conservation constraints O(I,J) for I=1..NX, J=1..NY (equality)
    # v = DX*U(I,J) - DX*U(I-1,J) + DY*V(I,J) - DY*V(I,J-1) - S*T = 0
    # SCALE = DX → identity constraint: v = 0 (scale doesn't change zeros)
    cons_data = [(i = i, j = j,
                  uij = u_idx(i, j), uim1j = u_idx(i-1, j),
                  vij = v_idx(i, j), vijm1 = v_idx(i, j-1))
                 for i in 1:NX for j in 1:NY]
    ExaModels.@add_con(c,
        DX * x[d.uij] - DX * x[d.uim1j] + DY * x[d.vij] - DY * x[d.vijm1] - S * x[1]
        for d in cons_data)

    # Capacity constraints (L type): U(I,J)^2 + ... <= 1
    cap_data = [(i = i, j = j,
                 pij = u_idx(i, j), pim1j = u_idx(i-1, j),
                 qij = v_idx(i, j), qijm1 = v_idx(i, j-1))
                for i in 1:NX for j in 1:NY]
    # A(I,J): U(I,J)^2 + V(I,J)^2 - 1 <= 0
    ExaModels.@add_con(c, x[d.pij]^2 + x[d.qij]^2 - T(1)
                       for d in cap_data; lcon = T(-Inf), ucon = T(0))
    # B(I,J): U(I-1,J)^2 + V(I,J)^2 - 1 <= 0
    ExaModels.@add_con(c, x[d.pim1j]^2 + x[d.qij]^2 - T(1)
                       for d in cap_data; lcon = T(-Inf), ucon = T(0))
    # C(I,J): U(I,J)^2 + V(I,J-1)^2 - 1 <= 0
    ExaModels.@add_con(c, x[d.pij]^2 + x[d.qijm1]^2 - T(1)
                       for d in cap_data; lcon = T(-Inf), ucon = T(0))
    # D(I,J): U(I-1,J)^2 + V(I,J-1)^2 - 1 <= 0
    ExaModels.@add_con(c, x[d.pim1j]^2 + x[d.qijm1]^2 - T(1)
                       for d in cap_data; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
