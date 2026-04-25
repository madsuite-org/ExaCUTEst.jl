# PENTAGON: 3-point pentagon problem, 6 variables, 15 inequality constraints
# Find 3 points in a 2D pentagon whose minimal distance is maximal.
#
# Variables: x[1]=X1, x[2]=Y1, x[3]=X2, x[4]=Y2, x[5]=X3, x[6]=Y3
# All free; start: X1=-1,Y1=0,X2=0,Y2=-1,X3=1,Y3=1
#
# Objective OBJ (N type, identity):
#   = 1/(D12^2)^8 + 1/(D13^2)^8 + 1/(D32^2)^8
#   where D12^2 = (X1-X2)^2 + (Y1-Y2)^2, etc.
#   IDIST element: DX = XA-XB, DY = YA-YB, F = 1/(DX^2+DY^2)^8
#
# Constraints C(I,J) for I=1..3, J=0..4 (all L type ≤ 0):
#   v = C(J)*X(I) + S(J)*Y(I) - 1 ≤ 0
#   where C(J) = cos(2π/5*J), S(J) = sin(2π/5*J), using SIF value 2π/5 = 1.2566371
#
# Constraint ordering: SIF order = I outer loop, J inner loop
#   C(1,0),C(1,1),...,C(1,4), C(2,0),...,C(2,4), C(3,0),...,C(3,4)
function pentagon_model(; T = Float64, backend = nothing, kwargs...)
    # Precompute cosine/sine coefficients using SIF value for 2π/5
    TwoPI5 = T(1.2566371)
    # cs[j+1] = (cos(TwoPI5*j), sin(TwoPI5*j)) for j=0..4
    cs = [(cos(TwoPI5 * T(j)), sin(TwoPI5 * T(j))) for j in 0:4]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6;
        lvar = T(-Inf), uvar = T(Inf),
        start = T[-1.0, 0.0, 0.0, -1.0, 1.0, 1.0])

    # Objective: sum of 1/((XA-XB)^2+(YA-YB)^2)^8 for the three pairs
    # D12: (X1,Y1)-(X2,Y2) = x[1,2] - x[3,4]
    ExaModels.@add_obj(c, T(1) / ((x[1] - x[3])^2 + (x[2] - x[4])^2)^8)
    # D13: (X1,Y1)-(X3,Y3) = x[1,2] - x[5,6]
    ExaModels.@add_obj(c, T(1) / ((x[1] - x[5])^2 + (x[2] - x[6])^2)^8)
    # D32: (X3,Y3)-(X2,Y2) = x[5,6] - x[3,4]
    ExaModels.@add_obj(c, T(1) / ((x[5] - x[3])^2 + (x[6] - x[4])^2)^8)

    # Constraints: C(I,J) = C(J)*X(I) + S(J)*Y(I) - 1 ≤ 0
    # Variable indices: X(1)=x[1], Y(1)=x[2], X(2)=x[3], Y(2)=x[4], X(3)=x[5], Y(3)=x[6]
    # Loop order: I=1..3 outer, J=0..4 inner → 15 constraints in SIF order
    ExaModels.@add_par(c, CJ, T[cs[j+1][1] for j in 0:4])
    ExaModels.@add_par(c, SJ, T[cs[j+1][2] for j in 0:4])

    # I=1: X(1)=x[1], Y(1)=x[2]
    ExaModels.@add_con(c, CJ[j] * x[1] + SJ[j] * x[2] - T(1) for j in 1:5;
        lcon = T(-Inf), ucon = T(0))
    # I=2: X(2)=x[3], Y(2)=x[4]
    ExaModels.@add_con(c, CJ[j] * x[3] + SJ[j] * x[4] - T(1) for j in 1:5;
        lcon = T(-Inf), ucon = T(0))
    # I=3: X(3)=x[5], Y(3)=x[6]
    ExaModels.@add_con(c, CJ[j] * x[5] + SJ[j] * x[6] - T(1) for j in 1:5;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
