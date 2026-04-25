# POLAK6: Nonlinear minmax in four variables (Polak, Mayne & Higgins 1991)
# Variation on ROSENMMX with chained Rosenbrock-like element types
# 5 free variables: X1..X4, U; start = 0
# EL42 = (X1-(X4+1)⁴)², EL442 = (X2-(X1-(X4+1)⁴)⁴)²
# EL4 = (X4+1)⁴, EL44 = (X1-(X4+1)⁴)⁴
function polak6_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 5; lvar = T(-Inf), uvar = T(Inf), start = T(0))
    ExaModels.@add_obj(c, x[5])
    # F1: rhs=0
    ExaModels.@add_con(c,
        -x[5] - T(5)*x[1] - T(5)*x[2] - T(21)*x[3] + T(7)*x[4]
        + (x[1] - (x[4]+T(1))^4)^2
        + (x[2] - (x[1]-(x[4]+T(1))^4)^4)^2
        + T(2)*x[3]^2 + x[4]^2
        + T(5)*(x[4]+T(1))^4
        + T(5)*(x[1]-(x[4]+T(1))^4)^4;
        lcon = T(-Inf), ucon = T(0))
    # F2: rhs=80
    ExaModels.@add_con(c,
        -x[5] + T(5)*x[1] - T(15)*x[2] - T(11)*x[3] - T(3)*x[4]
        + T(11)*(x[1]-(x[4]+T(1))^4)^2
        + T(11)*(x[2]-(x[1]-(x[4]+T(1))^4)^4)^2
        + T(12)*x[3]^2 + T(11)*x[4]^2
        - T(5)*(x[4]+T(1))^4
        + T(15)*(x[1]-(x[4]+T(1))^4)^4 - T(80);
        lcon = T(-Inf), ucon = T(0))
    # F3: rhs=100
    ExaModels.@add_con(c,
        -x[5] - T(15)*x[1] - T(5)*x[2] - T(21)*x[3] - T(3)*x[4]
        + T(11)*(x[1]-(x[4]+T(1))^4)^2
        + T(21)*(x[2]-(x[1]-(x[4]+T(1))^4)^4)^2
        + T(12)*x[3]^2 + T(21)*x[4]^2
        + T(15)*(x[4]+T(1))^4
        + T(5)*(x[1]-(x[4]+T(1))^4)^4 - T(100);
        lcon = T(-Inf), ucon = T(0))
    # F4: rhs=50
    ExaModels.@add_con(c,
        -x[5] + T(15)*x[1] - T(15)*x[2] - T(21)*x[3] - T(3)*x[4]
        + T(11)*(x[1]-(x[4]+T(1))^4)^2
        + T(11)*(x[2]-(x[1]-(x[4]+T(1))^4)^4)^2
        + T(12)*x[3]^2 + x[4]^2
        - T(15)*(x[4]+T(1))^4
        + T(15)*(x[1]-(x[4]+T(1))^4)^4 - T(50);
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
