# LUKVLE18: Chained modified HS53 problem
# Source: Problem 5.18, Luksan & Vlcek (1999), TR 767
# N=9997 variables (default), all free
# (N-1)/4 = 2499 blocks, NC = 3*(N-1)/4 = 7497 constraints
#
# With J = 4*(I-1) for I=1..(N-1)/4:
#   OBJ1(I): (x[J+1] - x[J+2])^4                (L4)
#   OBJ2(I): (x[J+2] + x[J+3] - 2)^2            (L2)
#   OBJ3(I): (x[J+4] - 1)^2                      (L2)
#   OBJ4(I): (x[J+5] - 1)^2                      (L2)
#
# Constraints (sequential j=1..NC; SIF RHS=0 for all constraint groups):
#   j≡1 mod 3: x[j]^2 + 3*x[j+1] = 0
#   j≡2 mod 3: x[j+1]^2 + x[j+2] - 2*x[j+3] = 0
#   j≡0 mod 3: x[j-1]^2 - x[j+2] = 0
#
# Start: all variables = 2.0
function lukvle18_model(N = 9997; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 1 "N must satisfy N ≡ 1 (mod 4) for LUKVLE18"
    NB = (N - 1) ÷ 4  # number of objective group blocks
    NC = 3 * NB        # total number of constraints

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(2))

    # Objective: with J+1=4I-3, J+2=4I-2, J+3=4I-1, J+4=4I, J+5=4I+1
    ExaModels.@add_obj(c, (x[4i-3] - x[4i-2])^4 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i-2] + x[4i-1] - T(2))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i] - T(1))^2 for i in 1:NB)
    ExaModels.@add_obj(c, (x[4i+1] - T(1))^2 for i in 1:NB)

    # Constraints in CUTEst sequential order j=1..NC (RHS=0, no constant subtraction):
    #   j≡1 mod 3: x[j]^2 + 3*x[j+1]
    #   j≡2 mod 3: x[j+1]^2 + x[j+2] - 2*x[j+3]
    #   j≡0 mod 3: x[j-1]^2 - x[j+2]
    #
    # j=1 (j≡1 mod 3): scalar to avoid j-1=0 index in j≡0 term
    ExaModels.@add_con(c, x[1]^2 + T(3) * x[2])

    # j=2..NC combined via selector arrays (i = j-1, j = i+1)
    ps1_data = T[((i + 1) % 3 == 1) ? T(1) : T(0) for i in 1:NC-1]
    ps2_data = T[((i + 1) % 3 == 2) ? T(1) : T(0) for i in 1:NC-1]
    ps3_data = T[((i + 1) % 3 == 0) ? T(1) : T(0) for i in 1:NC-1]

    ExaModels.@add_par(c, ps1, ps1_data)
    ExaModels.@add_par(c, ps2, ps2_data)
    ExaModels.@add_par(c, ps3, ps3_data)

    # j = i+1:
    #   ps1: x[i+1]^2 + 3*x[i+2]
    #   ps2: x[i+2]^2 + x[i+3] - 2*x[i+4]
    #   ps3: x[i]^2 - x[i+3]   (j≡0: x[j-1]^2 - x[j+2] with j-1=i, j+2=i+3)
    ExaModels.@add_con(c,
        ps1[i] * (x[i+1]^2 + T(3) * x[i+2]) +
        ps2[i] * (x[i+2]^2 + x[i+3] - T(2) * x[i+4]) +
        ps3[i] * (x[i]^2 - x[i+3])
        for i in 1:NC-1)

    return ExaModels.ExaModel(c; kwargs...)
end
