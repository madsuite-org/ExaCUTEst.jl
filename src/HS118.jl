function hs118_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # 15 variables with bounds: x1∈[8,21], x2∈[43,57], x3∈[3,16],
    # then groups of 3: x[3k+1]∈[0,90], x[3k+2]∈[0,120], x[3k+3]∈[0,60] for k=1..4
    ExaModels.@add_var(c, x, 15;
                       lvar = T[8,43,3, 0,0,0, 0,0,0, 0,0,0, 0,0,0],
                       uvar = T[21,57,16, 90,120,60, 90,120,60, 90,120,60, 90,120,60],
                       start = T[20,55,15, 20,60,20, 20,60,20, 20,60,20, 20,60,20])
    # Objective: sum_{k=0}^{4} [2.3*x[3k+1] + 1.7*x[3k+2] + 2.2*x[3k+3]
    #                            + 0.0001*x[3k+1]^2 + 0.0001*x[3k+2]^2 + 0.00015*x[3k+3]^2]
    ExaModels.@add_obj(c,
        T(2.3)*x[1]  + T(1.7)*x[2]  + T(2.2)*x[3]  + T(0.0001)*x[1]^2  + T(0.0001)*x[2]^2  + T(0.00015)*x[3]^2  +
        T(2.3)*x[4]  + T(1.7)*x[5]  + T(2.2)*x[6]  + T(0.0001)*x[4]^2  + T(0.0001)*x[5]^2  + T(0.00015)*x[6]^2  +
        T(2.3)*x[7]  + T(1.7)*x[8]  + T(2.2)*x[9]  + T(0.0001)*x[7]^2  + T(0.0001)*x[8]^2  + T(0.00015)*x[9]^2  +
        T(2.3)*x[10] + T(1.7)*x[11] + T(2.2)*x[12] + T(0.0001)*x[10]^2 + T(0.0001)*x[11]^2 + T(0.00015)*x[12]^2 +
        T(2.3)*x[13] + T(1.7)*x[14] + T(2.2)*x[15] + T(0.0001)*x[13]^2 + T(0.0001)*x[14]^2 + T(0.00015)*x[15]^2)
    # Constraints ordered as [A(K),B(K),C(K) for K=1..4], then D1..D5
    # ucon pattern: 13,13,14 repeated 4 times, then Inf×5
    # A(K): x[3K+1] - x[3K-2] + 7 in [0, 13]
    # B(K): x[3K+3] - x[3K]   + 7 in [0, 13]
    # C(K): x[3K+2] - x[3K-1] + 7 in [0, 14]
    # K=1
    ExaModels.@add_con(c, x[4]  - x[1]  + 7; lcon=T(0), ucon=T(13))  # A(1)
    ExaModels.@add_con(c, x[6]  - x[3]  + 7; lcon=T(0), ucon=T(13))  # B(1)
    ExaModels.@add_con(c, x[5]  - x[2]  + 7; lcon=T(0), ucon=T(14))  # C(1)
    # K=2
    ExaModels.@add_con(c, x[7]  - x[4]  + 7; lcon=T(0), ucon=T(13))  # A(2)
    ExaModels.@add_con(c, x[9]  - x[6]  + 7; lcon=T(0), ucon=T(13))  # B(2)
    ExaModels.@add_con(c, x[8]  - x[5]  + 7; lcon=T(0), ucon=T(14))  # C(2)
    # K=3
    ExaModels.@add_con(c, x[10] - x[7]  + 7; lcon=T(0), ucon=T(13))  # A(3)
    ExaModels.@add_con(c, x[12] - x[9]  + 7; lcon=T(0), ucon=T(13))  # B(3)
    ExaModels.@add_con(c, x[11] - x[8]  + 7; lcon=T(0), ucon=T(14))  # C(3)
    # K=4
    ExaModels.@add_con(c, x[13] - x[10] + 7; lcon=T(0), ucon=T(13))  # A(4)
    ExaModels.@add_con(c, x[15] - x[12] + 7; lcon=T(0), ucon=T(13))  # B(4)
    ExaModels.@add_con(c, x[14] - x[11] + 7; lcon=T(0), ucon=T(14))  # C(4)
    # D constraints: sum of group >= rhs (G type, no upper bound)
    ExaModels.@add_con(c, x[1]  + x[2]  + x[3]  - 60;  lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, x[4]  + x[5]  + x[6]  - 50;  lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, x[7]  + x[8]  + x[9]  - 70;  lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, x[10] + x[11] + x[12] - 85;  lcon=T(0), ucon=T(Inf))
    ExaModels.@add_con(c, x[13] + x[14] + x[15] - 100; lcon=T(0), ucon=T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
