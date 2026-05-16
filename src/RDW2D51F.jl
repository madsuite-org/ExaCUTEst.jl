# RDW2D51F/U/52B/F/U: Reaction-diffusion wave 2D PDE-constrained optimal control
# min H^2/36 * sum_{I,J=0}^{N-1} [M(U-V) + 2*beta * M0(F)]
# s.t. L(I,J) = 0 for interior nodes I,J=1..N-1 (FEM discrete Laplacian)
# Source: T. Rees, H.S. Dollar, A.J. Wathen, SIAM J. Sci. Comp. 2009
# SIF: QLR2-AN-V-V

# Linear index for F(I,J) and U(I,J), I,J=0..N
_rdw2d_fidx(I, J, N) = I * (N + 1) + J + 1
_rdw2d_uidx(I, J, N) = (N + 1)^2 + I * (N + 1) + J + 1

function _rdw2d_model(N::Int, beta::Float64,
                      Vfun,          # (I, J, N) → Float64
                      flb_fun,       # (I, J, N) → Float64 or nothing (no lb)
                      fub_fun,       # (I, J, N) → Float64 or nothing (no ub)
                      uib_fun;       # (I, J, N) → (lb, ub) or nothing (no bounds on U interior)
                      T = Float64, backend = nothing, kwargs...)
    Np  = N + 1
    NV  = 2 * Np^2
    H   = T(1) / T(N)
    H2o36 = H^2 / T(36)
    bH2o36 = T(2) * T(beta) * H2o36
    o6  = T(1) / T(6)
    H2o6 = H^2 / T(6)

    # Precompute V(I,J) for I,J=0..N
    V = Matrix{T}(undef, Np, Np)
    for I in 0:N, J in 0:N
        V[I+1, J+1] = T(Vfun(I, J, N))
    end

    # Bounds
    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)

    # U boundary: fixed to V(I,J)
    for J in 0:N
        ui = _rdw2d_uidx(0, J, N)
        lvar[ui] = V[1, J+1]; uvar[ui] = V[1, J+1]
        ui = _rdw2d_uidx(N, J, N)
        lvar[ui] = V[N+1, J+1]; uvar[ui] = V[N+1, J+1]
    end
    for I in 1:N-1
        ui = _rdw2d_uidx(I, 0, N)
        lvar[ui] = V[I+1, 1]; uvar[ui] = V[I+1, 1]
        ui = _rdw2d_uidx(I, N, N)
        lvar[ui] = V[I+1, N+1]; uvar[ui] = V[I+1, N+1]
    end

    # F boundary: fixed to 0
    for J in 0:N
        fi = _rdw2d_fidx(0, J, N); lvar[fi] = T(0); uvar[fi] = T(0)
        fi = _rdw2d_fidx(N, J, N); lvar[fi] = T(0); uvar[fi] = T(0)
    end
    for I in 1:N-1
        fi = _rdw2d_fidx(I, 0, N); lvar[fi] = T(0); uvar[fi] = T(0)
        fi = _rdw2d_fidx(I, N, N); lvar[fi] = T(0); uvar[fi] = T(0)
    end

    # F interior bounds
    if flb_fun !== nothing
        for I in 1:N-1, J in 1:N-1
            lvar[_rdw2d_fidx(I, J, N)] = T(flb_fun(I, J, N))
        end
    end
    if fub_fun !== nothing
        for I in 1:N-1, J in 1:N-1
            uvar[_rdw2d_fidx(I, J, N)] = T(fub_fun(I, J, N))
        end
    end

    # U interior bounds (for 52B)
    if uib_fun !== nothing
        for I in 1:N-1, J in 1:N-1
            lb, ub = uib_fun(I, J, N)
            ui = _rdw2d_uidx(I, J, N)
            lvar[ui] = T(lb); uvar[ui] = T(ub)
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = lvar, uvar = uvar, start = T(0))

    # Objective: sum over all N^2 elements
    # M element for (U-V): quadratic in U with known constants V
    # M0 element for F: quadratic in F
    elem_data = [
        (
            u1 = _rdw2d_uidx(I,   J,   N), v1 = V[I+1,   J+1],
            u2 = _rdw2d_uidx(I,   J+1, N), v2 = V[I+1,   J+2],
            u3 = _rdw2d_uidx(I+1, J,   N), v3 = V[I+2,   J+1],
            u4 = _rdw2d_uidx(I+1, J+1, N), v4 = V[I+2,   J+2],
            f1 = _rdw2d_fidx(I,   J,   N),
            f2 = _rdw2d_fidx(I,   J+1, N),
            f3 = _rdw2d_fidx(I+1, J,   N),
            f4 = _rdw2d_fidx(I+1, J+1, N),
        )
        for I in 0:N-1 for J in 0:N-1
    ]

    ExaModels.@add_obj(c,
        H2o36 * (
            T(2)*(x[d.u1]-d.v1)^2 + T(2)*(x[d.u2]-d.v2)^2 +
            T(2)*(x[d.u3]-d.v3)^2 + T(2)*(x[d.u4]-d.v4)^2 +
            T(2)*(x[d.u1]-d.v1)*(x[d.u2]-d.v2) +
            T(2)*(x[d.u1]-d.v1)*(x[d.u3]-d.v3) +
                 (x[d.u1]-d.v1)*(x[d.u4]-d.v4) +
                 (x[d.u2]-d.v2)*(x[d.u3]-d.v3) +
            T(2)*(x[d.u2]-d.v2)*(x[d.u4]-d.v4) +
            T(2)*(x[d.u3]-d.v3)*(x[d.u4]-d.v4)
        ) +
        bH2o36 * (
            T(2)*x[d.f1]^2 + T(2)*x[d.f2]^2 + T(2)*x[d.f3]^2 + T(2)*x[d.f4]^2 +
            T(2)*x[d.f1]*x[d.f2] + T(2)*x[d.f1]*x[d.f3] + x[d.f1]*x[d.f4] +
            x[d.f2]*x[d.f3] + T(2)*x[d.f2]*x[d.f4] + T(2)*x[d.f3]*x[d.f4]
        )
        for d in elem_data)

    # Constraints: L(I,J) = 0 for I,J = 1..N-1
    # 9-point stencil (derived from 4 surrounding bilinear elements):
    # (1/6) * [16*U(I,J) - 2*U(I±1,J) - 2*U(I,J±1) - 2*U(I+1,J+1) - 2*U(I-1,J-1)
    #          - 2*U(I+1,J-1) - 2*U(I-1,J+1)]
    # - (H^2/6) * [16*F(I,J) + 4*F(I±1,J) + 4*F(I,J±1) + F(I+1,J+1) + F(I-1,J-1)
    #              + F(I+1,J-1) + F(I-1,J+1)]
    con_data = [
        (
            u00  = _rdw2d_uidx(I,   J,   N),
            u01  = _rdw2d_uidx(I,   J+1, N),
            u0m1 = _rdw2d_uidx(I,   J-1, N),
            u10  = _rdw2d_uidx(I+1, J,   N),
            um10 = _rdw2d_uidx(I-1, J,   N),
            u11  = _rdw2d_uidx(I+1, J+1, N),
            u1m1 = _rdw2d_uidx(I+1, J-1, N),
            um11 = _rdw2d_uidx(I-1, J+1, N),
            um1m1= _rdw2d_uidx(I-1, J-1, N),
            f00  = _rdw2d_fidx(I,   J,   N),
            f01  = _rdw2d_fidx(I,   J+1, N),
            f0m1 = _rdw2d_fidx(I,   J-1, N),
            f10  = _rdw2d_fidx(I+1, J,   N),
            fm10 = _rdw2d_fidx(I-1, J,   N),
            f11  = _rdw2d_fidx(I+1, J+1, N),
            f1m1 = _rdw2d_fidx(I+1, J-1, N),
            fm11 = _rdw2d_fidx(I-1, J+1, N),
            fm1m1= _rdw2d_fidx(I-1, J-1, N),
        )
        for I in 1:N-1 for J in 1:N-1
    ]

    ExaModels.@add_con(c,
        o6 * (T(16)*x[d.u00]
              - T(2)*x[d.u01] - T(2)*x[d.u0m1]
              - T(2)*x[d.u10] - T(2)*x[d.um10]
              - T(2)*x[d.u11] - T(2)*x[d.u1m1]
              - T(2)*x[d.um11]- T(2)*x[d.um1m1])
        - H2o6 * (T(16)*x[d.f00]
                  + T(4)*x[d.f01] + T(4)*x[d.f0m1]
                  + T(4)*x[d.f10] + T(4)*x[d.fm10]
                  + x[d.f11] + x[d.f1m1] + x[d.fm11] + x[d.fm1m1])
        for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end

# V for 51x: (2*I/N - 1)^2 * (2*J/N - 1)^2 in lower-left quadrant [0,N/2]x[0,N/2]
_rdw2d51_V(I, J, N) = (I > N÷2 || J > N÷2) ? 0.0 : (2*I/N - 1)^2 * (2*J/N - 1)^2

# V for 52x: exp(-64*((I/N-0.5)^2 + (J/N-0.5)^2))
_rdw2d52_V(I, J, N) = exp(-64.0 * ((I/N - 0.5)^2 + (J/N - 0.5)^2))

# F lower bound for 51F and 52F/52B (from SIF BOUNDS section):
# ZL RDW2D51F F(I,J) = 0.1*(2-xi)*exp(-xi^2-xj^2)
_rdw2d51f_flb(I, J, N) = 0.1 * (2.0 - I/N) * exp(-(I/N)^2 - (J/N)^2)

# F upper bounds: 0.6 if J <= N/2, 0.9 if J > N/2
_rdw2d51f_fub(I, J, N) = J <= N÷2 ? 0.6 : 0.9

function rdw2d51f_model(N = 256; T = Float64, backend = nothing, kwargs...)
    _rdw2d_model(N, 0.01, _rdw2d51_V, _rdw2d51f_flb, _rdw2d51f_fub, nothing;
                 T = T, backend = backend, kwargs...)
end

function rdw2d51u_model(N = 256; T = Float64, backend = nothing, kwargs...)
    _rdw2d_model(N, 0.01, _rdw2d51_V, nothing, nothing, nothing;
                 T = T, backend = backend, kwargs...)
end

function rdw2d52b_model(N = 256; T = Float64, backend = nothing, kwargs...)
    # 52B: bounds on both F and U interior (same bounds for both from SIF)
    uib = (I, J, N) -> (_rdw2d51f_flb(I, J, N), _rdw2d51f_fub(I, J, N))
    _rdw2d_model(N, 0.005, _rdw2d52_V, _rdw2d51f_flb, _rdw2d51f_fub, uib;
                 T = T, backend = backend, kwargs...)
end

function rdw2d52f_model(N = 256; T = Float64, backend = nothing, kwargs...)
    _rdw2d_model(N, 0.005, _rdw2d52_V, _rdw2d51f_flb, _rdw2d51f_fub, nothing;
                 T = T, backend = backend, kwargs...)
end

function rdw2d52u_model(N = 256; T = Float64, backend = nothing, kwargs...)
    _rdw2d_model(N, 0.005, _rdw2d52_V, nothing, nothing, nothing;
                 T = T, backend = backend, kwargs...)
end
