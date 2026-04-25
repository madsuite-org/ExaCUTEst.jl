function bt11_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # 5 variables, all free, start = 2.0
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf), start = T(2.0))
    # OBJ: O1 + O2 + O3 + O4 + O5
    #   O1 = SSQ(X1, P=1) = (X1-1)^2
    #   O2 = ISQ(X1,X2) = (X1-X2)^2
    #   O3 = ISQ(X2,X3) = (X2-X3)^2
    #   O4 = IFR(X3,X4) = (X3-X4)^4
    #   O5 = IFR(X4,X5) = (X4-X5)^4
    ExaModels.@add_obj(c,
        (x[1] - T(1))^2 +
        (x[1] - x[2])^2 +
        (x[2] - x[3])^2 +
        (x[3] - x[4])^4 +
        (x[4] - x[5])^4
    )
    # Constants: C1 = sqrt(18) - 2, C2 = sqrt(8) - 2
    # CUTEst reorders: CON3 (linear) first, CON2 (1 element) second, CON1 (2 elements) last
    # CON3 (E): X1 - X5 - 2.0 = 0
    ExaModels.@add_con(c, x[1] - x[5] - T(2))
    # CON2 (E): X2 + X4 - X3^2 - C2 = 0
    ExaModels.@add_con(c, x[2] + x[4] - x[3]^2 - (sqrt(T(8)) - T(2)))
    # CON1 (E): X1 + X2^2 + X3^3 - C1 = 0
    ExaModels.@add_con(c, x[1] + x[2]^2 + x[3]^3 - (sqrt(T(18)) - T(2)))
    return ExaModels.ExaModel(c; kwargs...)
end
