# MINSURFO: minimal surface with obstacle (NX=50, NY=100 default)
# Variables: V(I,J) for I=0..NX+1, J=0..NY+1
# Objective: sum of two sets of groups, both of ROOT type: F=sqrt(alpha)
# Group A(I,J) for I=0..NX, J=0..NY, scale 1/AREA:
#   v = 1/HX^2*(V(I+1,J)-V(I,J))^2 + 1/HY^2*(V(I,J+1)-V(I,J))^2 - 1
#   contribution = (1/AREA) * sqrt(v/(1/AREA)) = sqrt(v*AREA) ... hmm
#   Actually: F(v/sigma)*sigma where sigma=1/AREA
#   = (1/AREA) * sqrt(v*AREA) = sqrt(v/AREA)
# Group B(I,J) for I=1..NX+1, J=1..NY+1, scale 1/AREA:
#   v = 1/HX^2*(V(I-1,J)-V(I,J))^2 + 1/HY^2*(V(I,J-1)-V(I,J))^2 - 1
# Bounds: V(0,J)=V(NX+1,J)=0; V(I,0)=V(I,NY+1)=1-(2I*HX-1)^2
#         Inner region: V(I,J) >= 1 for I in [floor(0.25/HX)..ceil(0.75/HX)], J in [...]
function minsurfo_model(NX = 50, NY = 100; T = Float64, backend = nothing, kwargs...)
    HX = T(1) / T(NX + 1)
    HY = T(1) / T(NY + 1)
    AREA = HX * HY * T(0.5)
    INV_HX2 = T(1) / HX^2
    INV_HY2 = T(1) / HY^2
    INV_AREA = T(1) / AREA

    # Variable index: V(I,J) for I=0..NX+1, J=0..NY+1
    # x[(I)*(NY+2) + (J+1)] using 1-based indexing
    idx(i, j) = i * (NY + 2) + j + 1
    N = (NX + 2) * (NY + 2)

    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start_vec = fill(T(0), N)

    # Boundary conditions
    for j in 0:NY+1
        lvar[idx(0, j)] = T(0); uvar[idx(0, j)] = T(0)
        lvar[idx(NX+1, j)] = T(0); uvar[idx(NX+1, j)] = T(0)
    end
    for i in 0:NX+1
        vij = T(1) - (T(2) * T(i) * HX - T(1))^2
        lvar[idx(i, 0)] = vij; uvar[idx(i, 0)] = vij; start_vec[idx(i, 0)] = vij
        lvar[idx(i, NY+1)] = vij; uvar[idx(i, NY+1)] = vij; start_vec[idx(i, NY+1)] = vij
    end

    # Lower bounds for inner obstacle region
    ix_lo = floor(Int, T(0.25) / HX)
    ix_hi = ceil(Int, T(0.75) / HX)
    iy_lo = floor(Int, T(0.25) / HY)
    iy_hi = ceil(Int, T(0.75) / HY)
    for i in ix_lo:min(ix_hi, NX+1)
        for j in iy_lo:min(iy_hi, NY+1)
            k = idx(i, j)
            if lvar[k] < T(1)
                lvar[k] = T(1)
            end
        end
    end

    # Start values
    for i in 0:NX+1
        for j in 0:NY+1
            k = idx(i, j)
            if start_vec[k] == T(0)
                start_vec[k] = T(1) - (T(2) * T(i) * HX - T(1))^2
            end
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start_vec)

    # Groups A(I,J) for I=0..NX, J=0..NY: scale=1/AREA, constant=-1
    # v_A = 1/HX^2*(V(I+1,J)-V(I,J))^2 + 1/HY^2*(V(I,J+1)-V(I,J))^2 + 1
    # F(v/sigma)*sigma = (1/AREA) * sqrt(v * AREA) = sqrt(v/AREA)
    grpA = [(i = i, j = j, k = idx(i, j), ki1j = idx(i+1, j),
             kij1 = idx(i, j+1)) for i in 0:NX for j in 0:NY]
    ExaModels.@add_obj(c,
        sqrt((INV_HX2 * (x[d.ki1j] - x[d.k])^2 + INV_HY2 * (x[d.kij1] - x[d.k])^2 + T(1)) * AREA)
        for d in grpA)

    # Groups B(I,J) for I=1..NX+1, J=1..NY+1: scale=1/AREA, constant=-1
    # v_B = 1/HX^2*(V(I-1,J)-V(I,J))^2 + 1/HY^2*(V(I,J-1)-V(I,J))^2 + 1
    grpB = [(i = i, j = j, k = idx(i, j), kim1j = idx(i-1, j),
             kijm1 = idx(i, j-1)) for i in 1:NX+1 for j in 1:NY+1]
    ExaModels.@add_obj(c,
        sqrt((INV_HX2 * (x[d.kim1j] - x[d.k])^2 + INV_HY2 * (x[d.kijm1] - x[d.k])^2 + T(1)) * AREA)
        for d in grpB)

    return ExaModels.ExaModel(c; kwargs...)
end
