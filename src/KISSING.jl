# KISSING: Kissing number problem - maximize minimum pairwise distance between NP points on a sphere
# Formulation: min Z  s.t. Z >= <x_i, x_j> for all i<j, ||x_i||^2 = 1 for all i
# NP=42 points in MDIM=3 dimensions
# Variables: X(i,j) for i=1..NP, j=1..MDIM and Z
# Variable ordering: X(1,1..3), X(2,1..3), ..., X(NP,1..3), Z
# Total variables: NP*MDIM + 1 = 42*3+1 = 127
# Constraints:
#   IC(i,j) for 1<=i<j<=NP: Z - sum_k X(i,k)*X(j,k) >= 0  (L type: v >= 0, lcon=0)
#   EC(i) for i=1..NP: sum_k X(i,k)^2 - 1 = 0  (E type equality)
function kissing_model(NP = 42, MDIM = 3; T = Float64, backend = nothing, kwargs...)
    # Variable index: X(i,j) → (i-1)*MDIM + j, Z → NP*MDIM + 1
    # Total vars = NP*MDIM + 1
    nvar = NP * MDIM + 1

    # Start point from SIF (provided for NP=42, MDIM=3):
    x0_data = T[
        -0.10890604,  0.85395078, -0.45461680,
         0.49883922, -0.18439316, -0.04798594,
         0.28262888, -0.48054070,  0.46715332,
        -0.00580106, -0.49987584, -0.44130302,
         0.81712540, -0.36874258, -0.68321896,
         0.29642426,  0.82315508,  0.35938150,
         0.09215152, -0.53564686,  0.00191436,
         0.11700318,  0.96722760, -0.14916438,
         0.01791524,  0.17759446, -0.61875872,
        -0.63833630,  0.80830972,  0.45846734,
         0.28446456,  0.45686938,  0.16368980,
         0.76557382,  0.16700944, -0.31647534,
    ]
    # Only first 12 points are specified in start; fill rest with zeros for NP>12
    start = zeros(T, nvar)
    for i in 1:min(NP, 12)
        for j in 1:MDIM
            start[(i-1)*MDIM + j] = x0_data[(i-1)*MDIM + j]
        end
    end

    # Pairs (i,j) with i < j for inequality constraints
    ineq_pairs = [(i = i, j = j) for i in 1:(NP-1) for j in (i+1):NP]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # X(i,k) are free, Z is free
    ExaModels.@add_var(c, x, nvar; lvar = T(-Inf), uvar = T(Inf), start = start)

    # Objective: minimize Z = x[NP*MDIM+1]
    ExaModels.@add_obj(c, x[NP * MDIM + 1])

    # Inequality: Z - <x_i, x_j> >= 0
    # x_i = x[(i-1)*MDIM+1 .. i*MDIM], x_j = x[(j-1)*MDIM+1 .. j*MDIM]
    # <x_i, x_j> = sum_{k=1}^MDIM x[(i-1)*MDIM+k] * x[(j-1)*MDIM+k]
    ExaModels.@add_con(c,
        x[NP * MDIM + 1] - sum(x[(d.i-1)*MDIM+k] * x[(d.j-1)*MDIM+k] for k in 1:MDIM)
        for d in ineq_pairs;
        lcon = T(0), ucon = T(Inf))

    # Equality: ||x_i||^2 = 1
    ExaModels.@add_con(c,
        sum(x[(i-1)*MDIM+k]^2 for k in 1:MDIM) - T(1)
        for i in 1:NP)

    return ExaModels.ExaModel(c; kwargs...)
end
