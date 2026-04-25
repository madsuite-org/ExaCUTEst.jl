# HS70: water-flow routing problem, n=4, ncon=1
#
# Source: problem 70 in Hock & Schittkowski (1981).
# SIF classification: SQR2-MN-4-1
#
# CUTEst variable ordering: x[1]=X4, x[2]=X3, x[3]=X2, x[4]=X1
# (X3 has upper bound 1.0 which appears at position 2 in CUTEst's uvar)
#
# Objective: sum_{i=1}^{19} (Y1(i,x) + Y2(i,x) - ydata[i])^2
#   where C[i] = {0.1,1,...,18}, ydata[i] as given
#   Y1 element (V1=X2=x[3], V2=X3=x[2], V3=X4=x[1]):
#     B = V2 + V3*(1-V2)
#     CI = C/7.658
#     P1 = 1/(1 + 1/(12*V1))
#     P5 = CI^(-1) * CI^V1 = CI^(V1-1)
#     P6 = exp(V1*(1 - CI*B))
#     Y1 = P1 * V2 * B^V1 * C4*sqrt(V1) * CI^(V1-1) * P6
#   Y2 element (V1=X1=x[4], V2=X3=x[2], V3=X4=x[1]):
#     Same B, but P3 = (B/V3)^V1, P6 = exp(V1*(1-CI*B/V3))
#     Y2 = P1*(1-V2)*(B/V3)^V1 * C4*sqrt(V1) * CI^(V1-1) * P6
#
# Constraint: C1 (G type) = X3 + X4 - X3*X4 ≥ 0
#   i.e. x[2] + x[1] - x[2]*x[1] ≥ 0  (lcon=0, ucon=Inf)
function hs70_model(; T = Float64, backend = nothing, kwargs...)
    # Problem constants
    C4 = T(sqrt(1.0 / 6.2832))  # = sqrt(1/(2*pi))
    C_data = T[0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
               10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]
    Y_data = T[0.00189, 0.1038, 0.268, 0.506, 0.577, 0.604, 0.725, 0.898, 0.947,
               0.845, 0.702, 0.528, 0.385, 0.257, 0.159, 0.0869, 0.0453, 0.01509, 0.00189]

    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    # x[1]=X4, x[2]=X3, x[3]=X2, x[4]=X1
    ExaModels.@add_var(c, x, 4;
                       lvar = T[1e-5, 1e-5, 1e-5, 1e-5],
                       uvar = T[100.0, 1.0, 100.0, 100.0],
                       start = T[2.0, 0.04, 4.0, 2.0])

    # Parameters: C and Y data (indexed by i=1..19)
    ExaModels.@add_par(c, pC, C_data)
    ExaModels.@add_par(c, pY, Y_data)

    # Objective: sum_{i=1..19} (Y1_i + Y2_i - Y_data[i])^2
    # Y1 element: V1=x[3](X2), V2=x[2](X3), V3=x[1](X4)
    # Y2 element: V1=x[4](X1), V2=x[2](X3), V3=x[1](X4)
    ExaModels.@add_obj(
        c,
        let V1a = x[3], V1b = x[4], V2 = x[2], V3 = x[1],
            B = V2 + V3 * (1 - V2),
            CI = pC[i] / T(7.658),
            P1a = 1 / (1 + 1 / (12 * V1a)),
            P1b = 1 / (1 + 1 / (12 * V1b)),
            P3a = B ^ V1a,
            P3b = (B / V3) ^ V1b,
            P4a = C4 * sqrt(V1a),
            P4b = C4 * sqrt(V1b),
            P5a = CI ^ (V1a - 1),
            P5b = CI ^ (V1b - 1),
            P6a = exp(V1a * (1 - CI * B)),
            P6b = exp(V1b * (1 - CI * B / V3)),
            Y1i = P1a * V2 * P3a * P4a * P5a * P6a,
            Y2i = P1b * (1 - V2) * P3b * P4b * P5b * P6b,
            t = Y1i + Y2i - pY[i]
            t * t
        end for i in 1:19
    )

    # Constraint C1: X3 + X4 - X3*X4 ≥ 0  (G type)
    # x[2]=X3, x[1]=X4
    ExaModels.@add_con(c, x[2] + x[1] - x[2] * x[1]; lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
