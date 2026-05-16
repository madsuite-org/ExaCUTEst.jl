# ROCKET: Maximize rocket altitude (COPS problem 10), NH=400 default
# Source: Dolan & More (2000); SIF input by N. Gould, 2000
#
# Variables: STEP, H(0..NH), V(0..NH), M(0..NH), T(0..NH), D(0..NH), G(0..NH)
# Julia: x[1]=STEP; for I=0..NH: H(I)=x[6I+2], V(I)=x[6I+3], M(I)=x[6I+4],
#                                  T(I)=x[6I+5], D(I)=x[6I+6], G(I)=x[6I+7]
#
# Objective: -H(NH) (minimize -H(NH) = maximize H(NH))
#
# Constraints for I=0..NH (E type):
#   D(I): -D(I) + DC * V(I)^2 * exp(HC*(1 - H(I))) = 0
#   G(I): -G(I) + G0 / H(I)^2 = 0
# For J=1..NH (E type):
#   H(J): -H(J) + H(J-1) + 0.5*STEP*(V(J)+V(J-1)) = 0
#   V(J): -V(J) + V(J-1) + 0.5*STEP*(T(J)-D(J))/M(J) - 0.5*STEP*G(J)
#                         + 0.5*STEP*(T(J-1)-D(J-1))/M(J-1) - 0.5*STEP*G(J-1) = 0
#   M(J): -M(J) + M(J-1) - (1/(2C))*STEP*(T(J)+T(J-1)) = 0
#           where C = sqrt(G0*H0)*0.5 = 0.5, so 1/(2C) = 1.0
#           → M(J) = M(J-1) - STEP*(T(J)+T(J-1))
#
# Bounds: STEP>=0, H(I)>=H0=1, V(I)>=0, T(I) in [0,TMAX], M(I) in [MF,M0]
#         H(0)=H0=1, V(0)=0, M(0)=1, M(NH)=MF=0.6 fixed
function rocket_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    G0   = T(1.0)
    H0   = T(1.0)
    M0   = T(1.0)
    TC   = T(3.5)
    HC   = T(500.0)
    VC   = T(620.0)
    MC   = T(0.6)
    MF   = MC * M0       # 0.6
    DC   = M0 / G0 * VC * T(0.5)  # 310.0
    TMAX = M0 * G0 * TC  # 3.5
    C    = sqrt(G0 * H0) * T(0.5)  # 0.5
    INV2C = T(1) / (T(2) * C)     # 1.0

    nv = 1 + 6 * (NH + 1)
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)
    start = zeros(T, nv)

    # STEP >= 0
    lvar[1] = T(0); start[1] = T(1) / T(NH)

    RNH = T(NH)
    for I in 0:NH
        base = 6 * I + 2
        # H(I) >= H0
        lvar[base] = H0; start[base] = H0
        # V(I) >= 0
        lvar[base + 1] = T(0)
        # M(I) in [MF, M0]
        lvar[base + 2] = MF; uvar[base + 2] = M0
        # T(I) in [0, TMAX]
        lvar[base + 3] = T(0); uvar[base + 3] = TMAX
        start[base + 3] = TMAX / T(2)
        # D(I): free (but should be positive)
        # G(I): free

        RI = T(I) / RNH
        RI1 = T(1) - RI
        start[base + 1] = RI * RI1          # V(I) = I/NH * (1-I/NH)
        start[base + 2] = MF + (M0 - MF) * (T(1) - RI)  # M(I) linear from M0 to MF

        # D(I) start: DC * V(I)^2 * exp(HC*(1-H(I)=1)) = 0 (since H(I)=1 start)
        # G(I) start: G0/H(I)^2 = G0
        start[base + 5] = G0
    end

    # Fixed boundary conditions
    lvar[2] = H0; uvar[2] = H0           # H(0) = H0 = 1
    lvar[3] = T(0); uvar[3] = T(0)      # V(0) = 0
    lvar[4] = M0; uvar[4] = M0; start[4] = M0  # M(0) = M0 = 1
    lvar[6 * NH + 4] = MF; uvar[6 * NH + 4] = MF  # M(NH) = MF = 0.6
    start[6 * NH + 4] = MF

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: -H(NH)  (maximize H(NH))
    ExaModels.@add_obj(c, -x[6 * NH + 2])

    # Constraints D(I) for I=0..NH: -D(I) + DC * V(I)^2 * exp(HC*(1-H(I))) = 0
    ExaModels.@add_con(c,
        -x[6 * d.I + 6] + DC * x[6 * d.I + 3]^2 * exp(HC * (T(1) - x[6 * d.I + 2]))
        for d in [(I = I,) for I in 0:NH])

    # Constraints G(I) for I=0..NH: -G(I) + G0/H(I)^2 = 0
    ExaModels.@add_con(c,
        -x[6 * d.I + 7] + G0 / x[6 * d.I + 2]^2
        for d in [(I = I,) for I in 0:NH])

    # Constraints H(J) for J=1..NH: -H(J) + H(J-1) + 0.5*STEP*(V(J)+V(J-1)) = 0
    ExaModels.@add_con(c,
        -x[6 * d.J + 2] + x[6 * (d.J - 1) + 2] +
        T(0.5) * x[1] * (x[6 * d.J + 3] + x[6 * (d.J - 1) + 3])
        for d in [(J = J,) for J in 1:NH])

    # Constraints V(J) for J=1..NH:
    # -V(J) + V(J-1) + 0.5*STEP*(T(J)-D(J))/M(J) - 0.5*STEP*G(J)
    #       + 0.5*STEP*(T(J-1)-D(J-1))/M(J-1) - 0.5*STEP*G(J-1) = 0
    ExaModels.@add_con(c,
        -x[6 * d.J + 3] + x[6 * (d.J - 1) + 3] +
        T(0.5) * x[1] * (x[6 * d.J + 5] - x[6 * d.J + 6]) / x[6 * d.J + 4] -
        T(0.5) * x[1] * x[6 * d.J + 7] +
        T(0.5) * x[1] * (x[6 * (d.J - 1) + 5] - x[6 * (d.J - 1) + 6]) / x[6 * (d.J - 1) + 4] -
        T(0.5) * x[1] * x[6 * (d.J - 1) + 7]
        for d in [(J = J,) for J in 1:NH])

    # Constraints M(J) for J=1..NH: -M(J) + M(J-1) - INV2C*STEP*(T(J)+T(J-1)) = 0
    ExaModels.@add_con(c,
        -x[6 * d.J + 4] + x[6 * (d.J - 1) + 4] -
        INV2C * x[1] * (x[6 * d.J + 5] + x[6 * (d.J - 1) + 5])
        for d in [(J = J,) for J in 1:NH])

    return ExaModels.ExaModel(c; kwargs...)
end
