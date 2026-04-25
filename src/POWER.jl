# POWER: Power problem by Oren (N=10000 variables, unconstrained)
# minimize (sum_{i=1}^N i * x_i^2)^2
# All variables free, starting at 1.0
# Source: Oren (1974), SIF: Toint, Dec 1989
# SIF: one L2 group G with N elements E(I)=SQ(X(I)), coefficient RI=I
#   v_G = sum_{i=1}^N i * x_i^2
#   contribution = v_G^2 = (sum_i i*x_i^2)^2
# Expansion: (sum_i i*x_i^2)^2 = sum_{i,j} i*j*x_i^2*x_j^2
# Default N=10000 is impractical for ExaModels (N^2=10^8 terms);
# use N=1000 and override CUTEst with -param N=1000.
function power_model(N = 1000; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1))
    ExaModels.@add_obj(c, i*j*x[i]^2*x[j]^2 for i in 1:N, j in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
