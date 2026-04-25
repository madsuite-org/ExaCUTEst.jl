# BROWNDEN: Brown and Dennis problem (4 variables, unconstrained least squares)
# 20 groups, all L2 type (no scale): contribution = v_i^2
# v_i = A_elem_i + B_elem_i
# A_elem_i = (X1 + (i/5)*X2 - exp(i/5))^2
# B_elem_i = (X3 + sin(i/5)*X4 - cos(i/5))^2
# BOUNDS: FR DEFAULT → all variables free
function brownden_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar  = T[-Inf, -Inf, -Inf, -Inf],
        uvar  = T[ Inf,  Inf,  Inf,  Inf],
        start = T[25.0, 5.0, -5.0, -1.0])

    # ti = i/5 for i = 1..20
    # A_elem_i: V1=X1, V2=X2, CV2=ti, CIND=exp(ti)  → (X1 + ti*X2 - exp(ti))^2
    # B_elem_i: V1=X3, V2=X4, CV2=sin(ti), CIND=cos(ti) → (X3 + sin(ti)*X4 - cos(ti))^2
    # group value v_i = A_elem_i + B_elem_i
    # contribution = v_i^2 (L2 group type, no scale)
    ExaModels.@add_obj(c,
        (  (x[1] + T(0.2)*i*x[2] - exp(T(0.2)*i))^2
         + (x[3] + sin(T(0.2)*i)*x[4] - cos(T(0.2)*i))^2)^2
        for i in 1:20)

    return ExaModels.ExaModel(c; kwargs...)
end
