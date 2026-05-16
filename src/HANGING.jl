# HANGING: 3D catenary hanging grid problem (Nesterova and Vial 1994)
# NX=40, NY=30 rectangular grid hung from 4 corners
# 3 variables per grid point: X(I,J), Y(I,J), Z(I,J)
# Variable index: X(I,J)→(i-1)*NY+j triplet → 3*((i-1)*NY+j-1)+1,+2,+3
# Objective: minimize sum Z(I,J) (gravity)
# Constraints:
#   RC(I,J) for I=1..NX, J=1..NY-1: (X-diff)^2 + (Y-diff)^2 + (Z-diff)^2 <= LX^2
#   DC(I,J) for I=1..NX-1, J=1..NY: (X-diff)^2 + (Y-diff)^2 + (Z-diff)^2 <= LY^2
# ISQ element: IV U = XX - YY, F = U^2 = (XX-YY)^2
# Fixed corner variables:
#   X(1,1)=0, Y(1,1)=0, Z(1,1)=0
#   X(NX,1)=NX, Y(NX,1)=0, Z(NX,1)=0
#   X(1,NY)=0, Y(1,NY)=NY, Z(1,NY)=0
#   X(NX,NY)=NX, Y(NX,NY)=NY, Z(NX,NY)=0
# LX=1.8, LY=1.8, LX^2=3.24, LY^2=3.24
# Start: X(I,J) = I-1, Y(I,J) = J-1, Z(I,J) = 0
function hanging_model(NX = 40, NY = 30; T = Float64, backend = nothing, kwargs...)
    LX = T(1.8)
    LY = T(1.8)
    LX2 = LX * LX
    LY2 = LY * LY

    nvar = 3 * NX * NY  # 3 variables per point: X,Y,Z in order

    # Variable index: var_idx(i,j,k) = 3*((i-1)*NY + (j-1)) + k
    # k=1: X, k=2: Y, k=3: Z
    vidx(i, j, k) = 3 * ((i - 1) * NY + (j - 1)) + k

    # Build lvar, uvar, start
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)

    for i in 1:NX
        for j in 1:NY
            start[vidx(i, j, 1)] = T(i - 1)
            start[vidx(i, j, 2)] = T(j - 1)
            start[vidx(i, j, 3)] = T(0)
        end
    end

    # Fix corner variables
    corners = [
        (1, 1, 1, T(0)), (1, 1, 2, T(0)), (1, 1, 3, T(0)),
        (NX, 1, 1, T(NX)), (NX, 1, 2, T(0)), (NX, 1, 3, T(0)),
        (1, NY, 1, T(0)), (1, NY, 2, T(NY)), (1, NY, 3, T(0)),
        (NX, NY, 1, T(NX)), (NX, NY, 2, T(NY)), (NX, NY, 3, T(0)),
    ]
    for (i, j, k, v) in corners
        idx = vidx(i, j, k)
        lvar[idx] = v
        uvar[idx] = v
        start[idx] = v
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum Z(I,J) = sum x[vidx(i,j,3)]
    ExaModels.@add_obj(c, x[d.idx] for d in [(idx = 3 * ((i - 1) * NY + (j - 1)) + 3,) for i in 1:NX for j in 1:NY])

    # RC(I,J) constraints: sum of squared diffs in x,y,z direction between (I,J) and (I,J+1)
    # (X(I,J+1)-X(I,J))^2 + (Y(I,J+1)-Y(I,J))^2 + (Z(I,J+1)-Z(I,J))^2 - LX^2 <= 0
    # i.e. residual >= -LX^2 is wrong; it's ≤ LX^2 → residual = sq_sum - LX^2, ucon=0
    rc_pairs = [(i = i, j = j) for i in 1:NX for j in 1:(NY-1)]
    ExaModels.@add_con(c,
        (x[3 * ((d.i - 1) * NY + d.j) + 1] - x[3 * ((d.i - 1) * NY + (d.j - 1)) + 1])^2 +
        (x[3 * ((d.i - 1) * NY + d.j) + 2] - x[3 * ((d.i - 1) * NY + (d.j - 1)) + 2])^2 +
        (x[3 * ((d.i - 1) * NY + d.j) + 3] - x[3 * ((d.i - 1) * NY + (d.j - 1)) + 3])^2 - LX2
        for d in rc_pairs;
        lcon = T(-Inf), ucon = T(0))

    # DC(I,J) constraints: sum of squared diffs between (I,J) and (I+1,J)
    dc_pairs = [(i = i, j = j) for i in 1:(NX-1) for j in 1:NY]
    ExaModels.@add_con(c,
        (x[3 * (d.i * NY + (d.j - 1)) + 1] - x[3 * ((d.i - 1) * NY + (d.j - 1)) + 1])^2 +
        (x[3 * (d.i * NY + (d.j - 1)) + 2] - x[3 * ((d.i - 1) * NY + (d.j - 1)) + 2])^2 +
        (x[3 * (d.i * NY + (d.j - 1)) + 3] - x[3 * ((d.i - 1) * NY + (d.j - 1)) + 3])^2 - LY2
        for d in dc_pairs;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
