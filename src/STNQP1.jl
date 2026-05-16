# STNQP1: Non-convex QP with binary-sum structure
# P=13, N=2^P=8192, NV=N+1=8193 variables, NC=4095 equality constraints
# Objective: sum_{I=0}^N x[I+1]^2 - 0.5*sum_{L=1}^{N/P} sum_{I=1}^P (sum_{j=0}^{I-1} x[2^j+L+1])^2
# Same constraints as STCQP1.
# Source: N. Gould, May 1996
function stnqp1_model(; T = Float64, backend = nothing, kwargs...)
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

    # Same constraints as STCQP1: E(I,L) for odd L, sum_{j=0}^{I-1} x[L+2^j] = I
    # CUTEst order: L outer, I inner — use masking coefficients for variable-length sums
    con_data = [
        (k1=Int32(l+1),   k2=Int32(l+2),   k3=Int32(l+4),   k4=Int32(l+8),
         k5=Int32(l+16),  k6=Int32(l+32),  k7=Int32(l+64),  k8=Int32(l+128),
         k9=Int32(l+256), k10=Int32(l+512), k11=Int32(l+1024), k12=Int32(l+2048),
         k13=Int32(l+4096),
         c2=T(i>=2), c3=T(i>=3), c4=T(i>=4), c5=T(i>=5), c6=T(i>=6), c7=T(i>=7),
         c8=T(i>=8), c9=T(i>=9), c10=T(i>=10), c11=T(i>=11), c12=T(i>=12), c13=T(i>=13),
         rhs=T(i))
        for l in 1:2:NP for i in 1:P
    ]
    ExaModels.@add_con(c,
        x[d.k1] + d.c2*x[d.k2] + d.c3*x[d.k3] + d.c4*x[d.k4] + d.c5*x[d.k5] +
        d.c6*x[d.k6] + d.c7*x[d.k7] + d.c8*x[d.k8] + d.c9*x[d.k9] + d.c10*x[d.k10] +
        d.c11*x[d.k11] + d.c12*x[d.k12] + d.c13*x[d.k13] - d.rhs
        for d in con_data)

    return ExaModels.ExaModel(c; kwargs...)
end
