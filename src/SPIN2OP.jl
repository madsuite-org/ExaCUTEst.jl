# SPIN2OP: Condensed spin equilibrium with objective X(1)^2, N=50 default
# Source: N. Trefethen; SIF input by N. Gould, 2009
# Same as SPIN2 but with objective: minimize X(1)^2
#
# CUTEst variable ordering (X(1) first due to OBJ element reference):
#   x[1]=X(1), x[2]=OMEGA, x[3]=MU, x[4]=Y(1)
#   X(I)=x[2*I+1] for I≥2, Y(I)=x[2*I+2] for I≥2
# Objective: X(1)^2 = x[1]^2
function spin2op_model(N = 50; T = Float64, backend = nothing, kwargs...)
    PI     = T(4) * atan(T(1))
    TWO_PI_N = T(2) * PI / T(N)

    xidx(I) = I == 1 ? 1 : 2*I + 1
    yidx(I) = I == 1 ? 4 : 2*I + 2

    nv = 2 + 2 * N
    startv = fill(T(1), nv)
    startv[xidx(1)] = cos(TWO_PI_N)
    startv[yidx(1)] = sin(TWO_PI_N)
    for I in 2:N
        startv[xidx(I)] = cos(TWO_PI_N * T(I))
        startv[yidx(I)] = sin(TWO_PI_N * T(I))
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = T(-Inf), uvar = T(Inf), start = startv)

    # Objective: X(1)^2 = x[1]^2
    ExaModels.@add_obj(c, x[1]^2)

    # R(I) constraints: -MU*X(I) + OMEGA*Y(I) + sum_{K≠I} (Y(I)-Y(K))/denom = 0
    # MU=x[3], OMEGA=x[2]
    for I in 1:N
        xi = xidx(I)
        yi = yidx(I)
        others = [k for k in 1:N if k != I]
        xk = [xidx(k) for k in others]
        yk = [yidx(k) for k in others]
        ExaModels.@add_con(c,
            -x[3] * x[xi] + x[2] * x[yi] +
            sum((x[yi] - x[yk[m]]) /
                ((x[yi] - x[yk[m]])^2 + (x[xi] - x[xk[m]])^2)
                for m in 1:length(others)))
    end
    # I(I) constraints: -MU*Y(I) - OMEGA*X(I) - sum_{K≠I} (X(I)-X(K))/denom = 0
    for I in 1:N
        xi = xidx(I)
        yi = yidx(I)
        others = [k for k in 1:N if k != I]
        xk = [xidx(k) for k in others]
        yk = [yidx(k) for k in others]
        ExaModels.@add_con(c,
            -x[3] * x[yi] - x[2] * x[xi] -
            sum((x[xi] - x[xk[m]]) /
                ((x[xi] - x[xk[m]])^2 + (x[yi] - x[yk[m]])^2)
                for m in 1:length(others)))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
