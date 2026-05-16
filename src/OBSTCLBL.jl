# OBSTCLBL: Quadratic obstacle problem (Dembo and Tulowitzki)
# Problem B, starting point L: lower obstacle = (sin(9.2*xi1)*sin(9.3*xi2))^3
# upper obstacle = lower^2 + 0.02; start from lower
function obstclbl_model(PX = 100, PY = 100; T = Float64, backend = nothing, kwargs...)
    PXm1 = PX - 1
    PYm1 = PY - 1
    HX = T(1.0 / PXm1)
    HY = T(1.0 / PYm1)
    HXHY = HX * HY
    C = T(1.0)
    LC = -HXHY * C
    HY4HX = HY / (T(4) * HX)
    HX4HY = HX / (T(4) * HY)

    idx(I, J) = (J - 1) * PY + I
    N = PX * PY

    lvar = fill(T(0), N)
    uvar = fill(T(Inf), N)
    start = fill(T(0), N)

    # Fix boundary nodes
    for J in 1:PX
        lvar[idx(1, J)] = T(0); uvar[idx(1, J)] = T(0)
        lvar[idx(PY, J)] = T(0); uvar[idx(PY, J)] = T(0)
    end
    for I in 2:PY-1
        lvar[idx(I, PX)] = T(0); uvar[idx(I, PX)] = T(0)
        lvar[idx(I, 1)] = T(0); uvar[idx(I, 1)] = T(0)
    end

    # Lower and upper obstacles for interior nodes
    for I in 2:PY-1
        xi1 = T((I - 1) / PYm1)
        sxi1 = sin(T(9.2) * xi1)
        for J in 2:PX-1
            xi2 = T((J - 1) / PXm1)
            sxi2 = sin(T(9.3) * xi2)
            L1 = sxi1 * sxi2
            L2 = L1 * L1
            low = L2 * L1   # L1^3
            upp = L2 + T(0.02)
            lvar[idx(I, J)] = low
            uvar[idx(I, J)] = upp
            start[idx(I, J)] = low  # starting point L
        end
    end

    interior = [(i = I, j = J) for J in 2:PX-1 for I in 2:PY-1]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_obj(c, LC * x[idx(d.i, d.j)] for d in interior)
    ExaModels.@add_obj(c, HY4HX * (x[idx(d.i+1, d.j)] - x[idx(d.i, d.j)])^2 for d in interior)
    ExaModels.@add_obj(c, HY4HX * (x[idx(d.i-1, d.j)] - x[idx(d.i, d.j)])^2 for d in interior)
    ExaModels.@add_obj(c, HX4HY * (x[idx(d.i, d.j+1)] - x[idx(d.i, d.j)])^2 for d in interior)
    ExaModels.@add_obj(c, HX4HY * (x[idx(d.i, d.j-1)] - x[idx(d.i, d.j)])^2 for d in interior)

    return ExaModels.ExaModel(c; kwargs...)
end
