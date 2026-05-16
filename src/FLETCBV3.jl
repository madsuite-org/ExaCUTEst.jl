# FLETCBV3: Boundary value problem (Fletcher 1992, scaled variant)
# Source: R. Fletcher, Numerical Analysis report NA/145, Dundee 1992
# N=5000 variables (free), start X(I)=I/(N+1)
#
# GROUP TYPE HALFL2(v, P): F = 0.5*P*v^2
# ELEMENT COS(V, P): F = P*cos(V)
#
# Parameters: OBJSCALE=1e8, P=1/OBJSCALE=1e-8, KAPPA=1.0
#   H=1/(N+1), H2=H^2, half_p=0.5*P
#   coeff_lin = P*(-1 - 2/H2)
#   coeff_cos = -KAPPA/H2 * P
#
# Objective:
#   sum_{I=0}^{N} HALFL2 groups (G(0)=0.5P*x[1]^2, G(I)=0.5P*(x[i]-x[i+1])^2, G(N)=0.5P*x[N]^2)
#   + sum_{I=1}^{N} coeff_lin * x[i]          (L(I) groups, linear identity)
#   + sum_{I=1}^{N} coeff_cos * P * cos(x[i])  (C(I) groups: element P*cos(X(I)) scaled by -KAPPA/H2)
function fletcbv3_model(N = 5000; T = Float64, backend = nothing, kwargs...)
    kappa    = T(1)
    p        = T(1e-8)
    rn1      = T(N + 1)
    h        = T(1) / rn1
    h2       = h^2
    half_p   = T(0.5) * p
    coeff_l  = p * (T(1) + T(2) / h2)  # P*-1-2/H2 = (1+2/H^2)*P > 0 (SIF param name is misleading)
    coeff_c  = -kappa / h2 * p  # = -KAPPA/H2*P = element contribution per term

    start = T[T(i) * h for i in 1:N]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)
    ExaModels.@add_obj(c, half_p * x[1]^2)
    ExaModels.@add_obj(c, half_p * (x[i] - x[i + 1])^2 for i in 1:(N - 1))
    ExaModels.@add_obj(c, half_p * x[N]^2)
    ExaModels.@add_obj(c, coeff_l * x[i] for i in 1:N)
    # C(I) element: P*cos(X(I)) added to C(I) group with coefficient -KAPPA/H2
    ExaModels.@add_obj(c, coeff_c * cos(x[i]) for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
