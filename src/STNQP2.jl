# STNQP2: Non-convex QP with binary-sum structure (variant 2 constraints)
# P=13, N=2^P=8192, NV=N+1=8193 variables, NC=4095 equality constraints
# Objective: same as STNQP1 (sum x[I+1]^2 - 0.5*sum(sum)^2)
# Constraints: same as STCQP2 (consecutive index pattern)
# Source: N. Gould, May 1996
function stnqp2_model(; T = Float64, backend = nothing, kwargs...)
    P  = 13
    N  = 2^P          # 8192
    NP = N ÷ P        # 630
    NV = N + 1        # 8193

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV;
        lvar = T(-2), uvar = T(2), start = T(0.5))

    # O(I) groups: PSQR with P=1.0 → x[I+1]^2
    ExaModels.@add_obj(c, x[i]^2 for i in 1:NV)

    # N(I,L) groups: PSQR with P=-0.5 → -0.5 * (sum)^2
    ExaModels.@add_obj(c, T(-0.5) * x[l + 2]^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l + 2] + x[l + 3])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l + 2] + x[l + 3] + x[l + 5])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2] + x[l+3] + x[l+5] + x[l+9])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65]+x[l+129])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65]+x[l+129]+x[l+257])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65]+x[l+129]+x[l+257]+x[l+513])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65]+x[l+129]+x[l+257]+x[l+513]+x[l+1025])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65]+x[l+129]+x[l+257]+x[l+513]+x[l+1025]+x[l+2049])^2 for l in 1:NP)
    ExaModels.@add_obj(c, T(-0.5) * (x[l+2]+x[l+3]+x[l+5]+x[l+9]+x[l+17]+x[l+33]+x[l+65]+x[l+129]+x[l+257]+x[l+513]+x[l+1025]+x[l+2049]+x[l+4097])^2 for l in 1:NP)

    # Constraints E(I,L) for odd L in 1..NP, I=1..P
    # E(I,L): sum_{J=1}^{I} x[(L-1)*P + J + 1] = I
    # CUTEst order: L outer, I inner — use masking coefficients for variable-length sums
    con_data = [
        (b=Int32((l-1)*P + 1),
         c2=T(i>=2), c3=T(i>=3), c4=T(i>=4), c5=T(i>=5), c6=T(i>=6), c7=T(i>=7),
         c8=T(i>=8), c9=T(i>=9), c10=T(i>=10), c11=T(i>=11), c12=T(i>=12), c13=T(i>=13),
         rhs=T(i))
        for l in 1:2:NP for i in 1:P
    ]
    ExaModels.@add_con(c,
        x[d.b+1] + d.c2*x[d.b+2] + d.c3*x[d.b+3] + d.c4*x[d.b+4] + d.c5*x[d.b+5] +
        d.c6*x[d.b+6] + d.c7*x[d.b+7] + d.c8*x[d.b+8] + d.c9*x[d.b+9] + d.c10*x[d.b+10] +
        d.c11*x[d.b+11] + d.c12*x[d.b+12] + d.c13*x[d.b+13] - d.rhs
        for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end
