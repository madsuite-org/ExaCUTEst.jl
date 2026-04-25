function hs113_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # CUTEst variable ordering: x[1..10] with SIF X4↔X9 swapped
    # CUTEst: [X1,X2,X3,X9_sif,X5,X6,X7,X8,X4_sif,X10], all free (FR DEFAULT)
    ExaModels.@add_var(c, x, 10;
                       lvar = T[-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf],
                       uvar = T[Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf,Inf],
                       start = T[2.0, 3.0, 5.0, 6.0, 1.0, 2.0, 7.0, 3.0, 5.0, 10.0])
    # Objective (identity group type):
    # 0.5*x'Hx + c'x + 1352
    # Quadratic part: x1^2 + x1*x2 + x2^2 + x3^2 + 2x4^2 + x5^2 + 2x6^2 + 5x7^2 + 7x8^2 + 4x9^2 + x10^2
    # Linear part: -14x1 - 16x2 - 20x3 - 40x4 - 6x5 - 4x6 - 154x8 - 40x9 - 14x10
    ExaModels.@add_obj(c,
        x[1]^2 + x[1]*x[2] + x[2]^2 + x[3]^2 + 2*x[4]^2 + x[5]^2 + 2*x[6]^2 +
        5*x[7]^2 + 7*x[8]^2 + 4*x[9]^2 + x[10]^2 -
        14*x[1] - 16*x[2] - 20*x[3] - 40*x[4] - 6*x[5] - 4*x[6] -
        154*x[8] - 40*x[9] - 14*x[10] + 1352)
    # C1: -4X1-5X2+3X7-9X8+105 >= 0
    ExaModels.@add_con(c, -4*x[1] - 5*x[2] + 3*x[7] - 9*x[8] + 105; lcon=T(0), ucon=T(Inf))
    # C2: -10X1+8X2+17X7-2X8 >= 0
    ExaModels.@add_con(c, -10*x[1] + 8*x[2] + 17*x[7] - 2*x[8]; lcon=T(0), ucon=T(Inf))
    # C3: 8X1-2X2-5X9_sif+2X10+12 >= 0  (X9_sif = x[4])
    ExaModels.@add_con(c, 8*x[1] - 2*x[2] - 5*x[4] + 2*x[10] + 12; lcon=T(0), ucon=T(Inf))
    # C4: 12X1+24X2+7X4_sif+72-3X1^2-4X2^2-2X3^2 >= 0  (X4_sif = x[9])
    ExaModels.@add_con(c, 12*x[1] + 24*x[2] + 7*x[9] + 72 - 3*x[1]^2 - 4*x[2]^2 - 2*x[3]^2;
                       lcon=T(0), ucon=T(Inf))
    # C5: -8X2+12X3+2X4_sif+4-5X1^2-X3^2 >= 0  (X4_sif = x[9])
    ExaModels.@add_con(c, -8*x[2] + 12*x[3] + 2*x[9] + 4 - 5*x[1]^2 - x[3]^2;
                       lcon=T(0), ucon=T(Inf))
    # C6: 8X1+16X2+X6-34-0.5X1^2-2X2^2-3X5^2 >= 0
    ExaModels.@add_con(c, 8*x[1] + 16*x[2] + x[6] - 34 - T(0.5)*x[1]^2 - 2*x[2]^2 - 3*x[5]^2;
                       lcon=T(0), ucon=T(Inf))
    # C7: 8X2-14X5+6X6-8-X1^2-2X2^2+2X1X2 >= 0
    ExaModels.@add_con(c, 8*x[2] - 14*x[5] + 6*x[6] - 8 - x[1]^2 - 2*x[2]^2 + 2*x[1]*x[2];
                       lcon=T(0), ucon=T(Inf))
    # C8: 3X1-6X2+192X9_sif+7X10-768-12X9_sif^2 >= 0  (X9_sif = x[4])
    ExaModels.@add_con(c, 3*x[1] - 6*x[2] + 192*x[4] + 7*x[10] - 768 - 12*x[4]^2;
                       lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
