# SREADIN3: Optimal control problem from tide modelling (OOR2-MN-V-V)
# Source: Nick Gould, July 1991 (Nancy Nichols)
# N=2000 discretization points. NV = 2*(N+1) = 4002. NC = N+1 = 2001.
# Variables: X(0)..X(N) (state), U(0)..U(N) (control), interleaved
# X in [-0.5, 0.5], U in [0, 1], start = 0.25
# Objective: trapezoidal integration of ENERGY(I) = (H/2)*U(I)*(X(I)-cos(2πtI))²
# Constraints: ODE discretization + periodic boundary condition
function sreadin3_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    A = T(0.07716)
    inv2A = T(1) / (T(2) * A)
    PI2 = T(2) * T(π)
    H = T(1) / T(N)
    H2 = H / T(2)

    NV = 2 * (N + 1)
    xi(i) = 2 * i + 1
    ui(i) = 2 * i + 2

    lvar = Vector{T}(undef, NV)
    uvar = Vector{T}(undef, NV)
    x0 = fill(T(0.25), NV)
    for i in 0:N
        lvar[xi(i)] = T(-0.5); uvar[xi(i)] = T(0.5)
        lvar[ui(i)] = T(0);    uvar[ui(i)] = T(1)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = x0)

    obj_data = [(
        xi  = Int32(xi(i)),   ui  = Int32(ui(i)),
        xi1 = Int32(xi(i-1)), ui1 = Int32(ui(i-1)),
        ci  = T(cos(PI2 * i * H)),
        ci1 = T(cos(PI2 * (i-1) * H)),
        h2  = H2,
    ) for i in 1:N]
    ExaModels.@add_obj(c,
        -d.h2 * x[d.ui] * (x[d.xi] - d.ci)^2
        - d.h2 * x[d.ui1] * (x[d.xi1] - d.ci1)^2
        for d in obj_data)

    con_data = [(
        xi  = Int32(xi(i)),   ui  = Int32(ui(i)),
        xi1 = Int32(xi(i-1)), ui1 = Int32(ui(i-1)),
        invH = T(1) / H,
        cc_i  = T(cos(PI2 * i * H)) * (-inv2A),
        cc_i1 = T(cos(PI2 * (i-1) * H)) * (-inv2A),
        p = inv2A,
    ) for i in 1:N]
    ExaModels.@add_con(c,
        d.invH * (x[d.xi] - x[d.xi1])
        + d.cc_i * x[d.ui] + d.cc_i1 * x[d.ui1]
        + d.p * x[d.ui] * x[d.xi] + d.p * x[d.ui1] * x[d.xi1]
        for d in con_data)

    period_data = [(x0 = Int32(xi(0)), xN = Int32(xi(N)), invH = T(1) / H)]
    ExaModels.@add_con(c, (x[d.x0] - x[d.xN]) * d.invH for d in period_data)

    return ExaModels.ExaModel(c; kwargs...)
end
