# OBSTCLAL: Quadratic obstacle problem (Dembo and Tulowitzki)
# Problem A: force field with sinusoidal lower obstacle, upper bound 2000
# PX=PY=100, n=10000 free interior variables
# Variables indexed (I,J) stored as x[(J-1)*PY+I] in column-major order
# SIF variable X(I,J): row I in 1..PY, col J in 1..PX
# ISQ element: F = (V1-V2)^2, IV U = V1 - V2
# Objective: sum over interior nodes G(I,J) = LC*x(I,J) + HY/4HX*(A+C) + HX/4HY*(B+D)
#   where A=(x(I+1,J)-x(I,J))^2, B=(x(I,J+1)-x(I,J))^2, etc.
function obstclal_model(PX = 100, PY = 100; T = Float64, backend = nothing, kwargs...)
    PXm1 = PX - 1
    PYm1 = PY - 1
    HX = T(1.0 / PXm1)
    HY = T(1.0 / PYm1)
    HXHY = HX * HY
    C = T(1.0)
    LC = -HXHY * C
    HY4HX = HY / (T(4) * HX)
    HX4HY = HX / (T(4) * HY)

    # Variable layout: x[idx(I,J)] where idx(I,J) = (J-1)*PY + I, I in 1..PY, J in 1..PX
    idx(I, J) = (J - 1) * PY + I

    N = PX * PY

    # Build lower obstacle values: lvar for interior nodes (I,J) with I in 2..PY-1, J in 2..PX-1
    lvar = fill(T(0), N)
    uvar = fill(T(2000), N)
    start = fill(T(0), N)

    # Fix boundary nodes
    for J in 1:PX
        lvar[idx(1, J)] = T(0)
        uvar[idx(1, J)] = T(0)
        lvar[idx(PY, J)] = T(0)
        uvar[idx(PY, J)] = T(0)
    end
    for I in 2:PY-1
        lvar[idx(I, PX)] = T(0)
        uvar[idx(I, PX)] = T(0)
        lvar[idx(I, 1)] = T(0)
        uvar[idx(I, 1)] = T(0)
    end

    # Set lower obstacle for interior nodes
    for I in 2:PY-1
        xsi1 = T((I - 1) / PYm1)
        sxsi1 = sin(T(3.2) * xsi1)
        for J in 2:PX-1
            xsi2 = T((J - 1) / PXm1)
            sxsi2 = sin(T(3.3) * xsi2)
            low = sxsi1 * sxsi2
            lvar[idx(I, J)] = low
            start[idx(I, J)] = low  # start from lower obstacle
        end
    end

    # Interior nodes data
    interior = [(i = I, j = J) for J in 2:PX-1 for I in 2:PY-1]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Objective: for each interior node G(I,J) = LC*x(I,J) + nonlinear contributions
    # LC*x(I,J) part
    ExaModels.@add_obj(c, LC * x[idx(d.i, d.j)] for d in interior)

    # HY/4HX * [(x(I+1,J)-x(I,J))^2 + (x(I-1,J)-x(I,J))^2]
    ExaModels.@add_obj(c, HY4HX * (x[idx(d.i+1, d.j)] - x[idx(d.i, d.j)])^2 for d in interior)
    ExaModels.@add_obj(c, HY4HX * (x[idx(d.i-1, d.j)] - x[idx(d.i, d.j)])^2 for d in interior)
    # HX/4HY * [(x(I,J+1)-x(I,J))^2 + (x(I,J-1)-x(I,J))^2]
    ExaModels.@add_obj(c, HX4HY * (x[idx(d.i, d.j+1)] - x[idx(d.i, d.j)])^2 for d in interior)
    ExaModels.@add_obj(c, HX4HY * (x[idx(d.i, d.j-1)] - x[idx(d.i, d.j)])^2 for d in interior)

    return ExaModels.ExaModel(c; kwargs...)
end
