function hs7_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T[-Inf, -Inf], uvar = T[Inf, Inf],
                       start = T[2.0, 2.0])
    # OBJ: X2 coeff=-1, element LN(X1)=log(1+x[1]²)
    ExaModels.@add_obj(c, log(T(1) + x[1]^2) - x[2])
    # CON1 (E): SQ+1(X1)=(1+x[1]²)², SQ(X2)=x[2]², constant=4
    ExaModels.@add_con(c, (T(1) + x[1]^2)^2 + x[2]^2 - T(4))
    return ExaModels.ExaModel(c; kwargs...)
end
