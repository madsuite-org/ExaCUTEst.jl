# HIMMELBB: Himmelblau problem 27 (2 free vars, start=[-1.2,1.0])
# Element H(X1,X2): R1=X1*X2, R2=1-X1, R3=1-X2-X1*R2^5 → F=R1*R2*R3
# Group G (L2): contribution = F^2 = (X1*X2*(1-X1)*(1-X2-X1*(1-X1)^5))^2
function himmelbb_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[T(-1.2), T(1.0)])
    ExaModels.@add_obj(c, let R2 = T(1) - x[1]
        (x[1] * x[2] * R2 * (T(1) - x[2] - x[1] * R2^5))^2
    end)
    return ExaModels.ExaModel(c; kwargs...)
end
