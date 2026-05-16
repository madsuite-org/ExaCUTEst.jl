function trigger_model(; T = Float64, backend = nothing, kwargs...)
    # TRIGGER: circuit equations, 7 variables (X1-X5, OUTPUT, INPUT=fixed)
    # 6 equality constraints
    # Variables: X1(1), X2(2), X3(3), X4(4), X5(5), OUTPUT(6), INPUT(7)
    # INPUT is fixed: lvar=uvar=INVALUE=0.322866124

    R0 = 10000.0; R1 = 39.0; R2 = 51.0; R3 = 10.0
    R4 = 25.5;   R5 = 1.0;  R6 = 0.62; R7 = 13.0; R8 = 0.201

    A11 = 1/R0 + 1/R1 + 1/R2
    A12 = -1/R1
    A22 = 1/R1 + 1/R3
    A26 = -1/R3
    A31 = -1/R0
    A33 = 1/R0 + 1/R4
    A34 = -1/R4
    A44 = 1/R4 + 1/R5 + 1/R6
    A45 = -1/R5
    A55 = 1/R5 + 1/R7
    A56 = -1/R7
    A66 = 1/R3 + 1/R7 + 1/R8
    inv_R2  = 1/R2
    inv_R8  = 1/R8
    INVALUE = 0.322866124

    B1 = 5.6e-8
    B2 = 1962.0

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    # CUTEst variable ordering: X5 (used in DIODE element) appears before X4 (not in elements)
    # pos 1=X1, 2=X2, 3=X3, 4=X5, 5=X4, 6=OUTPUT, 7=INPUT
    ExaModels.@add_var(c, x, 7;
        lvar = T[-Inf, -Inf, -Inf, -Inf, -Inf, -Inf, INVALUE],
        uvar = T[Inf, Inf, Inf, Inf, Inf, Inf, INVALUE],
        start = T[0.2, 0.6, 0.2, 0.6, 0.2, 9.6, INVALUE])

    # CUTEst order: linear groups (E1,E4,E3) first, then nonlinear (E2,E5,E6)
    # Variable mapping: X4=x[5], X5=x[4] (swapped from declaration order)

    # E1: A11*X1 + A12*X2 + A31*X3 + (1/R2)*INPUT = 0
    ExaModels.@add_con(c,
        T(A11)*x[1] + T(A12)*x[2] + T(A31)*x[3] + T(inv_R2)*x[7])

    # E4: A34*X3 + A44*X4 + A45*X5 = 0  (X4=x[5], X5=x[4])
    ExaModels.@add_con(c,
        T(A34)*x[3] + T(A44)*x[5] + T(A45)*x[4])

    # E3: A31*X1 + A33*X3 + A34*X4 = 0  (X4=x[5])
    ExaModels.@add_con(c,
        T(A31)*x[1] + T(A33)*x[3] + T(A34)*x[5])

    # E2: A12*X1 + A22*X2 + A26*OUTPUT + DIODE(X2) = 0
    # DIODE(X) = B1*exp(25*X) - B1
    ExaModels.@add_con(c,
        T(A12)*x[1] + T(A22)*x[2] + T(A26)*x[6]
        + T(B1)*exp(T(25)*x[2]) - T(B1))

    # E5: A45*X4 + A55*X5 + A56*OUTPUT + DIODE(X5) = 0  (X4=x[5], X5=x[4])
    ExaModels.@add_con(c,
        T(A45)*x[5] + T(A55)*x[4] + T(A56)*x[6]
        + T(B1)*exp(T(25)*x[4]) - T(B1))

    # E6: A26*X2 + A56*X5 + A66*OUTPUT + (-1/R8)*OPAMP(X1,X3) = 0
    # OPAMP: U = -X1+X3, F = 7.65*atan(B2*U)  (X5=x[4])
    ExaModels.@add_con(c,
        T(A26)*x[2] + T(A56)*x[4] + T(A66)*x[6]
        + T(-inv_R8) * T(7.65) * atan(T(B2)*(x[3] - x[1])))

    return ExaModels.ExaModel(c; kwargs...)
end
