# HART6: Hartman test problem in 6 variables (Dixon & Szego 1975)
# 6 variables ∈ [0,1]; start = 0.2
# Objective: -sum_{I=1..4} C(I)*exp(-sum_{J=1..6} A(I,J)*(X(J)-P(I,J))²)
# GROUP TYPE NEXP with SCALE=-1: contribution = CI*exp(-v)/(-1) = -CI*exp(-v)
function hart6_model(; T = Float64, backend = nothing, kwargs...)
    C = T[1.0, 1.2, 3.0, 3.2]
    A = T[10.0 0.05 17.0  3.5  1.7  8.0;
           0.05 10.0 17.0  0.1  8.0 14.0;
           3.0   3.5  1.7 10.0 17.0  8.0;
          17.0   8.0  0.05 10.0  0.1 14.0]
    P = T[0.1312 0.1696 0.5569 0.0124 0.8283 0.5886;
          0.2329 0.4135 0.8307 0.3736 0.1004 0.9991;
          0.2348 0.1451 0.3522 0.2883 0.3047 0.6650;
          0.4047 0.8828 0.8732 0.5743 0.1091 0.0381]
    data = [(ci=C[i],
             a1=A[i,1], a2=A[i,2], a3=A[i,3], a4=A[i,4], a5=A[i,5], a6=A[i,6],
             p1=P[i,1], p2=P[i,2], p3=P[i,3], p4=P[i,4], p5=P[i,5], p6=P[i,6])
            for i in 1:4]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6; lvar = T(0), uvar = T(1), start = T(0.2))
    ExaModels.@add_obj(c,
        -d.ci * exp(-(d.a1*(x[1]-d.p1)^2 + d.a2*(x[2]-d.p2)^2 + d.a3*(x[3]-d.p3)^2
                    + d.a4*(x[4]-d.p4)^2 + d.a5*(x[5]-d.p5)^2 + d.a6*(x[6]-d.p6)^2))
        for d in data)
    return ExaModels.ExaModel(c; kwargs...)
end
