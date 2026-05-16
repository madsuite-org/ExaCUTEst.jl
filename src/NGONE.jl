# NGONE: maximum area polygon, N=200 default (HNS=99)
# Variables: X(0)..X(HNS), Y(0)..Y(HNS) — 2*(HNS+1) total
# Ordering: X(0),Y(0),X(1),Y(1),...,X(HNS),Y(HNS) → x[2i+1]=X(i), x[2i+2]=Y(i)
# Objective: -0.5 * sum_{i=0}^{HNS-1} (X(i+1)-X(i))*(Y(i+1)+Y(i)) = -0.5*DX*SY
#   element 2PR2I: DX=XA-XB, SY=YA+YB; F=DX*SY
#   So area = -0.5*sum C(i) = -0.5*sum (X(i+1)-X(i))*(Y(i+1)+Y(i))
# Ordering constraints: X(i+1) >= X(i) for i=1..HNS-1 (G type, lcon=0)
# Distance constraints: (X(i)-X(j))^2 + (Y(i)-Y(j))^2 <= 1 for 0<=j<i<=HNS
# Fixed: X(0)=0, Y(0)=0, Y(HNS)=0; X(HNS)<=1 (UB)
function ngone_model(HNS = 99; T = Float64, backend = nothing, kwargs...)
    # Variables: x[2i+1] = X(i), x[2i+2] = Y(i) for i=0..HNS
    N = 2 * (HNS + 1)
    lvar = fill(T(-Inf), N)
    uvar = fill(T(Inf), N)
    start = fill(T(0), N)

    # Fix X(0)=0, Y(0)=0
    lvar[1] = T(0); uvar[1] = T(0)  # X(0)
    lvar[2] = T(0); uvar[2] = T(0)  # Y(0)
    # X(HNS) <= 1; start=2 (as in SIF)
    uvar[2*HNS+1] = T(1)
    start[2*HNS+1] = T(2)
    # Fix Y(HNS)=0
    lvar[2*HNS+2] = T(0); uvar[2*HNS+2] = T(0)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = lvar, uvar = uvar, start = start)

    # Helper: X(i) = x[2i+1], Y(i) = x[2i+2]
    # Objective: AREA = sum_{i=0}^{HNS-1} -0.5*(X(i+1)-X(i))*(Y(i+1)+Y(i))
    area_data = [(xi = 2*i+1, yi = 2*i+2, xi1 = 2*i+3, yi1 = 2*i+4) for i in 0:HNS-1]
    ExaModels.@add_obj(c, -T(0.5) * (x[d.xi1] - x[d.xi]) * (x[d.yi1] + x[d.yi]) for d in area_data)

    # Ordering constraints: X(i+1) >= X(i) for i=1..HNS-1: X(i+1) - X(i) >= 0
    ord_data = [(xi = 2*i+1, xi1 = 2*i+3) for i in 1:HNS-1]
    ExaModels.@add_con(c, x[d.xi1] - x[d.xi] for d in ord_data; lcon = T(0), ucon = T(Inf))

    # Distance constraints: (X(i)-X(j))^2 + (Y(i)-Y(j))^2 - 1 <= 0
    dist_data = [(xi = 2*i+1, yi = 2*i+2, xj = 2*j+1, yj = 2*j+2)
                 for i in 0:HNS for j in 0:i-1]
    ExaModels.@add_con(c, (x[d.xi] - x[d.xj])^2 + (x[d.yi] - x[d.yj])^2 - T(1)
                       for d in dist_data; lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
