function hs100_model(; T = Float64, backend = nothing, kwargs...)
    # CUTEst ordering: x[1]=X1, x[2]=X2, x[3]=X3, x[4]=X4, x[5]=X6, x[6]=X5, x[7]=X7
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7;
                       lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf],
                       uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, Inf],
                       start = T[1.0, 2.0, 0.0, 4.0, 1.0, 0.0, 1.0])
    # O1: L2 group (φ(v)=v²), v=X1-10 → (X1-10)²
    ExaModels.@add_obj(c, (x[1] - T(10))^2)
    # O2: L2 group SCALE=0.2, v=X2-12 → (X2-12)²/0.2 = 5*(X2-12)²
    ExaModels.@add_obj(c, T(5) * (x[2] - T(12))^2)
    # O4: L2 group SCALE=1/3, v=X4-11 → (X4-11)²/(1/3) = 3*(X4-11)²
    ExaModels.@add_obj(c, T(3) * (x[4] - T(11))^2)
    # O5: identity group; X5=x[6], X6=x[5], X3=x[3], X7=x[7]
    # elements: 10*X5^6 + X3^4 + 7*X6^2 + X7^4 - 4*X6*X7 - 10*X6 - 8*X7
    ExaModels.@add_obj(c, T(10)*x[6]^6 + x[3]^4 + T(7)*x[5]^2 + x[7]^4 - T(4)*x[5]*x[7] - T(10)*x[5] - T(8)*x[7])
    # C1 (G): -X3-5*X5+127-2*X1²-3*X2^4-4*X4² ≥ 0  (X5=x[6])
    ExaModels.@add_con(c, -x[3] - T(5)*x[6] + T(127) - T(2)*x[1]^2 - T(3)*x[2]^4 - T(4)*x[4]^2;
                       lcon = T(0), ucon = T(Inf))
    # C2 (G): -7*X1-3*X2-X4+X5+282-10*X3² ≥ 0  (X5=x[6])
    ExaModels.@add_con(c, -T(7)*x[1] - T(3)*x[2] - x[4] + x[6] + T(282) - T(10)*x[3]^2;
                       lcon = T(0), ucon = T(Inf))
    # C3 (G): -23*X1+8*X7+196-X2²-6*X6² ≥ 0  (X6=x[5])
    ExaModels.@add_con(c, -T(23)*x[1] + T(8)*x[7] + T(196) - x[2]^2 - T(6)*x[5]^2;
                       lcon = T(0), ucon = T(Inf))
    # C4 (G): -5*X6+11*X7-4*X1²-X2²+3*X1*X2-2*X3² ≥ 0  (X6=x[5])
    ExaModels.@add_con(c, -T(5)*x[5] + T(11)*x[7] - T(4)*x[1]^2 - x[2]^2 + T(3)*x[1]*x[2] - T(2)*x[3]^2;
                       lcon = T(0), ucon = T(Inf))
    return ExaModels.ExaModel(c; kwargs...)
end
