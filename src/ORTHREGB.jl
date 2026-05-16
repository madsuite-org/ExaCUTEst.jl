# ORTHREGB: Orthogonal regression - fit ellipse to 6 points in 3D
# 27 variables: H11,H12,H13,H22,H23,H33,G1,G2,G3 (9 params) + X(I),Y(I),Z(I) for I=1..6
# Objective: sum_I [(X-XD)^2 + (Y-YD)^2 + (Z-ZD)^2]
# Constraints E(I): H11*X^2 + 2*H12*X*Y + H22*Y^2 + 2*H13*X*Z + 2*H23*Y*Z + H33*Z^2
#                   - 2*G1*X - 2*G2*Y - 2*G3*Z = 1
function orthregb_model(; T = Float64, backend = nothing, kwargs...)
    A_val = 9.0; B_val = 6.0; C_val = 7.0
    CX = 0.5; CY = 0.5; CZ = 0.5

    NPTS = 6
    XD = T[CX+A_val, CX+B_val, CX-A_val, CX-B_val, CX, CX]
    YD = T[CY+A_val, CY-B_val, CY-A_val, CY+B_val, CY, CY]
    ZD = T[CZ, CZ, CZ, CZ, CZ+C_val, CZ-C_val]

    # Variables: H11(1),H12(2),H13(3),H22(4),H23(5),H33(6),G1(7),G2(8),G3(9)
    #            X(1..6)(10..15), Y(1..6)(16..21), Z(1..6)(22..27)
    nvar = 9 + 3 * NPTS
    idx_H11=1; idx_H12=2; idx_H13=3; idx_H22=4; idx_H23=5; idx_H33=6
    idx_G1=7; idx_G2=8; idx_G3=9
    idx_X(i) = 9 + i
    idx_Y(i) = 9 + NPTS + i
    idx_Z(i) = 9 + 2*NPTS + i

    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    start = zeros(T, nvar)
    start[idx_H11] = T(1.0); start[idx_H22] = T(1.0); start[idx_H33] = T(1.0)
    for I in 1:NPTS
        start[idx_X(I)] = XD[I]; start[idx_Y(I)] = YD[I]; start[idx_Z(I)] = ZD[I]
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    for I in 1:NPTS
        ExaModels.@add_obj(c, (x[idx_X(I)] - XD[I])^2 + (x[idx_Y(I)] - YD[I])^2 + (x[idx_Z(I)] - ZD[I])^2)
        ExaModels.@add_con(c,
            x[idx_H11] * x[idx_X(I)]^2 +
            T(2) * x[idx_H12] * x[idx_X(I)] * x[idx_Y(I)] +
            x[idx_H22] * x[idx_Y(I)]^2 +
            T(2) * x[idx_H13] * x[idx_X(I)] * x[idx_Z(I)] +
            T(2) * x[idx_H23] * x[idx_Y(I)] * x[idx_Z(I)] +
            x[idx_H33] * x[idx_Z(I)]^2 -
            T(2) * x[idx_G1] * x[idx_X(I)] -
            T(2) * x[idx_G2] * x[idx_Y(I)] -
            T(2) * x[idx_G3] * x[idx_Z(I)] - T(1.0))
    end

    return ExaModels.ExaModel(c; kwargs...)
end
