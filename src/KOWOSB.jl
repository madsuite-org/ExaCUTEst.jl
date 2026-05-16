# KOWOSB: Kowalik-Osborne enzyme kinetics problem (More-Garbow-Hillstrom #15)
# 4 variables, 11 groups (nonlinear least squares)
# min sum_{i=1}^11 (x1*(u_i^2 + u_i*x2) / (u_i^2 + u_i*x3 + x4) - y_i)^2
# where y_i = observations, u_i = measurement values
# Variables: x1, x2, x3, x4 all free
# Start: x1=0.25, x2=0.39, x3=0.415, x4=0.39
function kowosb_model(; T = Float64, backend = nothing, kwargs...)
    # Observations y_i
    y = T[0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627,
          0.0456, 0.0342, 0.0323, 0.0235, 0.0246]
    # Measurement values u_i (from P line)
    u = T[4.0, 2.0, 1.0, 0.5, 0.25, 0.167, 0.125, 0.1, 0.0833, 0.0714, 0.0624]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(-Inf), uvar = T(Inf),
                       start = T[0.25, 0.39, 0.415, 0.39])

    # Element KWO(V1,V2,V3,V4; U): V1*(U^2+U*V2) / (U^2+U*V3+V4)
    # Group G(i): L2 type, linear const = -y_i, element E(i)=KWO with coeff 1
    # Group value v_i = KWO(x,u_i) - y_i
    # Contribution = v_i^2 (L2 group type: F(v)=v^2, no scale)
    ExaModels.@add_par(c, Y, y)
    ExaModels.@add_par(c, U, u)
    ExaModels.@add_obj(c,
        (x[1] * (U[i]^2 + U[i] * x[2]) / (U[i]^2 + U[i] * x[3] + x[4]) - Y[i])^2
        for i in 1:11)

    return ExaModels.ExaModel(c; kwargs...)
end
