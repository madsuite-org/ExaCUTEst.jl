# SIPOW3: Semi-infinite LP, unit square boundary approximation (Powell 1992)
# 4 free variables: X1..X4; start = [-0.1, 0, 0, 1.2]
# Objective: X4; M/2 G-type + M/2 L-type constraints
# Points (xi,eta) parametrize the boundary of the unit square [0,1]²
# G: X1 + xi*X2 + eta*X3 + X4 - xi²*eta ≥ 0
# L: X1 + xi*X2 + eta*X3       - xi²*eta ≤ 0
function sipow3_model(M = 2000; T = Float64, backend = nothing, kwargs...)
    M2 = M ÷ 2
    step = 8.0 / M
    # Compute (xi,eta) along boundary of [0,1]² in four segments
    xi  = [(j-1)*step for j in 1:M2]
    eta = zeros(M2)
    M8  = M ÷ 8
    M4  = M ÷ 4
    M38 = 3*(M ÷ 8)
    # Segment 1: left edge x=0, y goes 0→1 (J=1..M/8)
    for j in 1:M8
        eta[j] = xi[j]
        xi[j]  = 0.0
    end
    # Segment 2: top edge y=1, x goes 0→1 (J=M/8+1..M/4)
    for j in M8+1:M4
        eta[j] = 1.0
        xi[j]  = xi[j] - 1.0
    end
    # Segment 3: right edge x=1, y goes 0→1 (J=M/4+1..3M/8)
    for j in M4+1:M38
        eta[j] = xi[j] - 2.0
        xi[j]  = 1.0
    end
    # Segment 4: bottom edge y=0, x goes 0→1 (J=3M/8+1..M/2)
    for j in M38+1:M2
        xi[j]  = xi[j] - 3.0
        eta[j] = 0.0
    end
    data = [(xi = T(xi[j]), eta = T(eta[j]), rhs = T(xi[j]^2 * eta[j])) for j in 1:M2]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4; lvar = T(-Inf), uvar = T(Inf),
        start = T[T(-0.1), T(0), T(0), T(1.2)])
    ExaModels.@add_obj(c, x[4])
    ExaModels.@add_con(c, x[1] + d.xi*x[2] + d.eta*x[3] + x[4] - d.rhs
        for d in data; lcon = T(0), ucon = T(Inf))
    ExaModels.@add_con(c, x[1] + d.xi*x[2] + d.eta*x[3] - d.rhs
        for d in data; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
