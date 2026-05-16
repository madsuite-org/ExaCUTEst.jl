# AIRPORT: Airport localization problem in Brazil
# 42 cities (N=42), 84 variables (x(I),y(I) for I=1..N)
# Minimize sum_{i<j} [(x(i)-x(j))^2 + (y(i)-y(j))^2]
# Subject to: (x(I)-CX(I))^2 + (y(I)-CY(I))^2 <= R(I)  for I=1..N
# Bounds: -10 <= x(I), y(I) <= 10
# Source: Nabholz & Ehrhardt / Ph. Toint, November 1994
function airport_model(; T = Float64, backend = nothing, kwargs...)
    N = 42

    R  = T[0.09, 0.3, 0.09, 0.45, 0.5, 0.04, 0.1, 0.02, 0.02, 0.07,
           0.4, 0.045, 0.05, 0.056, 0.36, 0.08, 0.07, 0.36, 0.67, 0.38,
           0.37, 0.05, 0.4, 0.66, 0.05, 0.07, 0.08, 0.3, 0.31, 0.49,
           0.09, 0.46, 0.12, 0.07, 0.07, 0.09, 0.05, 0.13, 0.16, 0.46,
           0.25, 0.1]
    CX = T[-6.3, -7.8, -9.0, -7.2, -5.7, -1.9, -3.5, -0.5, 1.4, 4.0,
            2.1, 5.5, 5.7, 5.7, 3.8, 5.3, 4.7, 3.3, 0.0, -1.0,
           -0.4, 4.2, 3.2, 1.7, 3.3, 2.0, 0.7, 0.1, -0.1, -3.5,
           -4.0, -2.7, -0.5, -2.9, -1.2, -0.4, -0.1, -1.0, -1.7, -2.1,
           -1.8, 0.0]
    CY = T[8.0, 5.1, 2.0, 2.6, 5.5, 7.1, 5.9, 6.6, 6.1, 5.6,
           4.9, 4.7, 4.3, 3.6, 4.1, 3.0, 2.4, 3.0, 4.7, 3.4,
           2.3, 1.5, 0.5, -1.7, -2.0, -3.1, -3.5, -2.4, -1.3, 0.0,
          -1.7, -2.1, -0.4, -2.9, -3.4, -4.3, -5.2, -6.5, -7.5, -6.4,
          -5.1, 0.0]

    # Variable indexing: x(I) = variable 2I-1, y(I) = variable 2I
    NV = 2 * N

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, xv, NV;
        lvar = T(-10), uvar = T(10))

    # Objective: sum_{I<J} [(x(I)-x(J))^2 + (y(I)-y(J))^2]
    # Group OBJ1(I,J) = (xv[2I-1] - xv[2J-1])^2
    # Group OBJ2(I,J) = (xv[2I]   - xv[2J])^2
    obj_data = [(xi = 2*I-1, yi = 2*I, xj = 2*J-1, yj = 2*J)
                for I in 1:N-1 for J in I+1:N]
    ExaModels.@add_obj(c,
        (xv[d.xi] - xv[d.xj])^2 + (xv[d.yi] - xv[d.yj])^2
        for d in obj_data)

    # Constraints: (x(I)-CX(I))^2 + (y(I)-CY(I))^2 - R(I) <= 0
    ExaModels.@add_par(c, CXp, CX)
    ExaModels.@add_par(c, CYp, CY)
    ExaModels.@add_par(c, Rp, R)
    ExaModels.@add_con(c,
        (xv[2*i - 1] - CXp[i])^2 + (xv[2*i] - CYp[i])^2 - Rp[i]
        for i in 1:N;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
