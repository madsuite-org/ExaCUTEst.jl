# OBSTCLAE: quadratic obstacle problem, PX=PY=100 default
# Variables: X(I,J) for J=1..PX, I=1..PY (P^2 total)
# Objective: sum over interior nodes (2..PY-1, 2..PX-1) of:
#   HY/(4HX)*[(X(i+1,j)-X(i,j))^2 + (X(i-1,j)-X(i,j))^2] +
#   HX/(4HY)*[(X(i,j+1)-X(i,j))^2 + (X(i,j-1)-X(i,j))^2] +
#   LINC*X(i,j)
# Boundary: X(1,j)=X(PY,j)=X(i,PX)=X(i,1)=0 (fixed)
# Upper obstacle: X(i,j) <= 2000 (upper bound)
# Lower obstacle: X(i,j) >= sin(3.2*(i-1)*HY)*sin(3.3*(j-1)*HX) for interior
# HX = 1/(PX-1), HY = 1/(PY-1), C=1
function obstclae_model(PX = 100, PY = 100; T = Float64, backend = nothing, kwargs...)
    HX = T(1) / T(PX - 1)
    HY = T(1) / T(PY - 1)
    C = T(1)
    C0 = HX * HY * C
    LINC = -C0
    HY4HX = HY / (T(4) * HX)
    HX4HY = HX / (T(4) * HY)

    # Variable index: X(I,J) maps to linear index (J-1)*PY + I
    idx(i, j) = (j - 1) * PY + i
    N = PX * PY

    lvar = fill(T(-Inf), N)
    uvar = fill(T(2000), N)
    start = fill(T(0), N)

    # Fix boundary variables
    for j in 1:PX
        lvar[idx(1, j)] = T(0); uvar[idx(1, j)] = T(0)
        lvar[idx(PY, j)] = T(0); uvar[idx(PY, j)] = T(0)
    end
    for i in 2:PY-1
        lvar[idx(i, 1)] = T(0); uvar[idx(i, 1)] = T(0)
        lvar[idx(i, PX)] = T(0); uvar[idx(i, PX)] = T(0)
    end

    # Lower bounds for interior nodes
    for i in 2:PY-1
        for j in 2:PX-1
            xi1 = T(i - 1) * HY
            xi2 = T(j - 1) * HX
            low = sin(T(3.2) * xi1) * sin(T(3.3) * xi2)
            if low > lvar[idx(i, j)]
                lvar[idx(i, j)] = low
            end
        end
    end

    # Start: interior nodes at 1.0 (point E)
    for i in 2:PY-1
        for j in 2:PX-1
            start[idx(i, j)] = T(1)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Objective: linear term
    interior = [(i = i, j = j, k = idx(i, j)) for i in 2:PY-1 for j in 2:PX-1]
    ExaModels.@add_obj(c, LINC * x[d.k] for d in interior)

    # Quadratic terms: 4 neighbors per interior node
    int4 = [(k = idx(i, j), kp1 = idx(i+1, j), km1 = idx(i-1, j),
             kj1 = idx(i, j+1), kjm1 = idx(i, j-1))
            for i in 2:PY-1 for j in 2:PX-1]
    ExaModels.@add_obj(c,
        HY4HX * ((x[d.kp1] - x[d.k])^2 + (x[d.km1] - x[d.k])^2) +
        HX4HY * ((x[d.kj1] - x[d.k])^2 + (x[d.kjm1] - x[d.k])^2)
        for d in int4)

    return ExaModels.ExaModel(c; kwargs...)
end
