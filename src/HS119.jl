function hs119_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 16;
                       lvar = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       uvar = T[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                                5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                       start = T[10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                                 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    # Objective: sum_{(I,J): A(I,J)=1} P(I)*P(J), where P(i) = x[i]^2 + x[i] + 1
    # Diagonal terms A(I,I)=1 for all I
    ExaModels.@add_obj(c, (x[i]^2 + x[i] + T(1))^2 for i in 1:16)
    # Off-diagonal terms with A(I,J)=1
    ExaModels.@add_obj(c, (x[1]^2 + x[1] + T(1)) * (x[4]^2 + x[4] + T(1)))
    ExaModels.@add_obj(c, (x[1]^2 + x[1] + T(1)) * (x[7]^2 + x[7] + T(1)))
    ExaModels.@add_obj(c, (x[1]^2 + x[1] + T(1)) * (x[8]^2 + x[8] + T(1)))
    ExaModels.@add_obj(c, (x[1]^2 + x[1] + T(1)) * (x[16]^2 + x[16] + T(1)))
    ExaModels.@add_obj(c, (x[2]^2 + x[2] + T(1)) * (x[3]^2 + x[3] + T(1)))
    ExaModels.@add_obj(c, (x[2]^2 + x[2] + T(1)) * (x[7]^2 + x[7] + T(1)))
    ExaModels.@add_obj(c, (x[2]^2 + x[2] + T(1)) * (x[10]^2 + x[10] + T(1)))
    ExaModels.@add_obj(c, (x[3]^2 + x[3] + T(1)) * (x[7]^2 + x[7] + T(1)))
    ExaModels.@add_obj(c, (x[3]^2 + x[3] + T(1)) * (x[9]^2 + x[9] + T(1)))
    ExaModels.@add_obj(c, (x[3]^2 + x[3] + T(1)) * (x[10]^2 + x[10] + T(1)))
    ExaModels.@add_obj(c, (x[3]^2 + x[3] + T(1)) * (x[14]^2 + x[14] + T(1)))
    ExaModels.@add_obj(c, (x[4]^2 + x[4] + T(1)) * (x[7]^2 + x[7] + T(1)))
    ExaModels.@add_obj(c, (x[4]^2 + x[4] + T(1)) * (x[11]^2 + x[11] + T(1)))
    ExaModels.@add_obj(c, (x[4]^2 + x[4] + T(1)) * (x[15]^2 + x[15] + T(1)))
    ExaModels.@add_obj(c, (x[5]^2 + x[5] + T(1)) * (x[6]^2 + x[6] + T(1)))
    ExaModels.@add_obj(c, (x[5]^2 + x[5] + T(1)) * (x[10]^2 + x[10] + T(1)))
    ExaModels.@add_obj(c, (x[5]^2 + x[5] + T(1)) * (x[12]^2 + x[12] + T(1)))
    ExaModels.@add_obj(c, (x[5]^2 + x[5] + T(1)) * (x[16]^2 + x[16] + T(1)))
    ExaModels.@add_obj(c, (x[6]^2 + x[6] + T(1)) * (x[8]^2 + x[8] + T(1)))
    ExaModels.@add_obj(c, (x[6]^2 + x[6] + T(1)) * (x[15]^2 + x[15] + T(1)))
    ExaModels.@add_obj(c, (x[7]^2 + x[7] + T(1)) * (x[11]^2 + x[11] + T(1)))
    ExaModels.@add_obj(c, (x[7]^2 + x[7] + T(1)) * (x[13]^2 + x[13] + T(1)))
    ExaModels.@add_obj(c, (x[8]^2 + x[8] + T(1)) * (x[10]^2 + x[10] + T(1)))
    ExaModels.@add_obj(c, (x[8]^2 + x[8] + T(1)) * (x[15]^2 + x[15] + T(1)))
    ExaModels.@add_obj(c, (x[9]^2 + x[9] + T(1)) * (x[12]^2 + x[12] + T(1)))
    ExaModels.@add_obj(c, (x[9]^2 + x[9] + T(1)) * (x[16]^2 + x[16] + T(1)))
    ExaModels.@add_obj(c, (x[10]^2 + x[10] + T(1)) * (x[14]^2 + x[14] + T(1)))
    ExaModels.@add_obj(c, (x[11]^2 + x[11] + T(1)) * (x[13]^2 + x[13] + T(1)))
    ExaModels.@add_obj(c, (x[12]^2 + x[12] + T(1)) * (x[14]^2 + x[14] + T(1)))
    ExaModels.@add_obj(c, (x[13]^2 + x[13] + T(1)) * (x[14]^2 + x[14] + T(1)))
    # Constraints: G(i) = sum_j B(i,j)*x[j] - C(i) = 0
    ExaModels.@add_con(c, T(0.22)*x[1] + T(0.20)*x[2] + T(0.19)*x[3] + T(0.25)*x[4] + T(0.15)*x[5] + T(0.11)*x[6] + T(0.12)*x[7] + T(0.13)*x[8] + T(1.00)*x[9] - T(2.5))
    ExaModels.@add_con(c, T(-1.46)*x[1] + T(-1.30)*x[3] + T(1.82)*x[4] + T(-1.15)*x[5] + T(0.80)*x[7] + T(1.00)*x[10] - T(1.1))
    ExaModels.@add_con(c, T(1.29)*x[1] + T(-0.89)*x[2] + T(-1.16)*x[5] + T(-0.96)*x[6] + T(-0.49)*x[8] + T(1.00)*x[11] - T(-3.1))
    ExaModels.@add_con(c, T(-1.10)*x[1] + T(-1.06)*x[2] + T(0.95)*x[3] + T(-0.54)*x[4] + T(-1.78)*x[6] + T(-0.41)*x[7] + T(1.00)*x[12] - T(-3.5))
    ExaModels.@add_con(c, T(-1.43)*x[4] + T(1.51)*x[5] + T(0.59)*x[6] + T(-0.33)*x[7] + T(-0.43)*x[8] + T(1.00)*x[13] - T(1.3))
    ExaModels.@add_con(c, T(-1.72)*x[2] + T(-0.33)*x[3] + T(1.62)*x[5] + T(1.24)*x[6] + T(0.21)*x[7] + T(-0.26)*x[8] + T(1.00)*x[14] - T(2.1))
    ExaModels.@add_con(c, T(1.12)*x[1] + T(0.31)*x[4] + T(1.12)*x[7] + T(-0.36)*x[9] + T(1.00)*x[15] - T(2.3))
    ExaModels.@add_con(c, T(0.45)*x[2] + T(0.26)*x[3] + T(-1.10)*x[4] + T(0.58)*x[5] + T(-1.03)*x[7] + T(0.10)*x[8] + T(1.00)*x[16] - T(-1.5))
    return ExaModels.ExaModel(c; kwargs...)
end
