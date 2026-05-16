# NYSTROM5: Fifth-order Nystrom integration method coefficients
# NOR2-RY-18-20: 18 variables (3 fixed), 20 equality constraints
# Source: Chawla & Sharma, Computing 26:247-256, 1981.
# SIF: FR 'DEFAULT' + FX AL1=0, AL2=0.2, AL4=1.0
# Variables: A(1..4), B(1..4), AL(1..4), BE(2,1), BE(3,1), BE(3,2), BE(4,1..3)
# (18 total; AL1, AL2, AL4 fixed)
# Constraints are 20 polynomial equations (E type, identity group type, no scale)
# Element types: 2PR=X*Y, 3PR=X*Y*Z, 4PR=X*Y*Z*W, XY2=X*Y^2, XY3=X*Y^3, XY4=X*Y^4
# 2PRI2: (X)*(Y1+Y2), 2PRI3: (X)*(Y1+Y2+Y3)
# XY2I2: X*(Y1+Y2)^2, XY3 group uses sum of Y etc.

function nystrom5_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # Variable indices (1-based, CUTEst ordering from DO loop):
    # I=1: A(1)=1, B(1)=2, AL(1)=3
    # I=2: A(2)=4, B(2)=5, AL(2)=6, BE(2,1)=7
    # I=3: A(3)=8, B(3)=9, AL(3)=10, BE(3,1)=11, BE(3,2)=12
    # I=4: A(4)=13, B(4)=14, AL(4)=15, BE(4,1)=16, BE(4,2)=17, BE(4,3)=18
    # Fixed: AL1(3)=0, AL2(6)=0.2, AL4(15)=1.0; start point sets AL2=0.2, AL4=1.0

    ExaModels.@add_var(c, x, 18;
        lvar = T[-Inf, -Inf, 0.0, -Inf, -Inf, 0.2, -Inf,
                  -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, -Inf, 1.0, -Inf, -Inf, -Inf],
        uvar = T[Inf, Inf, 0.0, Inf, Inf, 0.2, Inf,
                  Inf, Inf, Inf, Inf, Inf, Inf, Inf, 1.0, Inf, Inf, Inf],
        start = T[0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )

    # Short aliases for readability:
    # A1=x[1], A2=x[4], A3=x[8], A4=x[13]
    # B1=x[2], B2=x[5], B3=x[9], B4=x[14]
    # AL1=x[3](=0), AL2=x[6](=0.2), AL3=x[10], AL4=x[15](=1)
    # BE21=x[7], BE31=x[11], BE32=x[12], BE41=x[16], BE42=x[17], BE43=x[18]

    # Constraint 3A: A1+A2+A3+A4 = 0.5
    ExaModels.@add_con(c, x[1] + x[4] + x[8] + x[13] - T(0.5))

    # Constraint 3B: A1*AL1 + A2*AL2 + A3*AL3 + A4*AL4 = 1/6
    ExaModels.@add_con(c, x[1]*x[3] + x[4]*x[6] + x[8]*x[10] + x[13]*x[15] - T(1.0/6.0))

    # Constraint 3C: A1*AL1^2 + A2*AL2^2 + A3*AL3^2 + A4*AL4^2 = 1/12
    ExaModels.@add_con(c, x[1]*x[3]^2 + x[4]*x[6]^2 + x[8]*x[10]^2 + x[13]*x[15]^2 - T(1.0/12.0))

    # Constraint 3D: A1*AL1^3 + A2*AL2^3 + A3*AL3^3 + A4*AL4^3 = 1/20
    ExaModels.@add_con(c, x[1]*x[3]^3 + x[4]*x[6]^3 + x[8]*x[10]^3 + x[13]*x[15]^3 - T(1.0/20.0))

    # Constraint 4A: A2*(BE21) + A3*(BE31+BE32) + A4*(BE41+BE42+BE43) = 1/6
    # Elements: 4AE1=A2*BE21, 4AE2=A3*(BE31+BE32), 4AE3=A4*(BE41+BE42+BE43)
    ExaModels.@add_con(c, x[4]*x[7] + x[8]*(x[11]+x[12]) + x[13]*(x[16]+x[17]+x[18]) - T(1.0/6.0))

    # Constraint 4B: A2*AL2*BE21 + A3*AL3*(BE31+BE32) + A4*AL4*(BE41+BE42+BE43) = 1/24
    ExaModels.@add_con(c, x[4]*x[6]*x[7] + x[8]*x[10]*(x[11]+x[12]) + x[13]*x[15]*(x[16]+x[17]+x[18]) - T(1.0/24.0))

    # Constraint 4C: A2*BE21*AL1 + A3*(BE31*AL1+BE32*AL2) + A4*(BE41*AL1+BE42*AL2+BE43*AL3)= 1/24
    # Elements: 4CE1=A2*BE21*AL1, 4CE2=A3*BE31*AL1, 4CE3=A3*BE32*AL2,
    #           4CE4=A4*BE41*AL1, 4CE5=A4*BE42*AL2, 4CE6=A4*BE43*AL3
    ExaModels.@add_con(c, x[4]*x[7]*x[3] + x[8]*x[11]*x[3] + x[8]*x[12]*x[6]
                        + x[13]*x[16]*x[3] + x[13]*x[17]*x[6] + x[13]*x[18]*x[10] - T(1.0/24.0))

    # Constraint 5A: B1+B2+B3+B4 = 1
    ExaModels.@add_con(c, x[2] + x[5] + x[9] + x[14] - T(1.0))

    # Constraint 5B: B1*AL1 + B2*AL2 + B3*AL3 + B4*AL4 = 1/2
    ExaModels.@add_con(c, x[2]*x[3] + x[5]*x[6] + x[9]*x[10] + x[14]*x[15] - T(0.5))

    # Constraint 5C: B1*AL1^2 + B2*AL2^2 + B3*AL3^2 + B4*AL4^2 = 1/3
    ExaModels.@add_con(c, x[2]*x[3]^2 + x[5]*x[6]^2 + x[9]*x[10]^2 + x[14]*x[15]^2 - T(1.0/3.0))

    # Constraint 5D: B1*AL1^3 + B2*AL2^3 + B3*AL3^3 + B4*AL4^3 = 1/4
    ExaModels.@add_con(c, x[2]*x[3]^3 + x[5]*x[6]^3 + x[9]*x[10]^3 + x[14]*x[15]^3 - T(0.25))

    # Constraint 5E: B1*AL1^4 + B2*AL2^4 + B3*AL3^4 + B4*AL4^4 = 1/5
    ExaModels.@add_con(c, x[2]*x[3]^4 + x[5]*x[6]^4 + x[9]*x[10]^4 + x[14]*x[15]^4 - T(0.2))

    # Constraint 6A: B2*BE21 + B3*(BE31+BE32) + B4*(BE41+BE42+BE43) = 1/6
    # Elements: B2BE21=B2*BE21, 6AE2=B3*(BE31+BE32), 6AE3=B4*(BE41+BE42+BE43)
    ExaModels.@add_con(c, x[5]*x[7] + x[9]*(x[11]+x[12]) + x[14]*(x[16]+x[17]+x[18]) - T(1.0/6.0))

    # Constraint 6B: B2*AL2*BE21 + B3*AL3*(BE31+BE32) + B4*AL4*(BE41+BE42+BE43) = 1/8
    ExaModels.@add_con(c, x[5]*x[6]*x[7] + x[9]*x[10]*(x[11]+x[12]) + x[14]*x[15]*(x[16]+x[17]+x[18]) - T(0.125))

    # Constraint 6C: B2*AL2^2*BE21 + B3*AL3^2*(BE31+BE32) + B4*AL4^2*(BE41+BE42+BE43) = 1/10
    ExaModels.@add_con(c, x[5]*x[6]^2*x[7] + x[9]*x[10]^2*(x[11]+x[12]) + x[14]*x[15]^2*(x[16]+x[17]+x[18]) - T(0.1))

    # Constraint 7: B2*BE21^2 + B3*(BE31+BE32)^2 + B4*(BE41+BE42+BE43)^2 = 1/20
    # Elements: 7E1=B2*BE21^2, 7E2=B3*(BE31+BE32)^2, 7E3=B4*(BE41+BE42+BE43)^2
    ExaModels.@add_con(c, x[5]*x[7]^2 + x[9]*(x[11]+x[12])^2 + x[14]*(x[16]+x[17]+x[18])^2 - T(1.0/20.0))

    # Constraint 8A: B2*BE21*AL1 + B3*(BE31*AL1 + BE32*AL2) + B4*(BE41*AL1 + BE42*AL2 + BE43*AL3) = 1/24
    ExaModels.@add_con(c, x[5]*x[7]*x[3] + x[9]*x[11]*x[3] + x[9]*x[12]*x[6]
                        + x[14]*x[16]*x[3] + x[14]*x[17]*x[6] + x[14]*x[18]*x[10] - T(1.0/24.0))

    # Constraint 8B: B2*BE21*AL2*AL1 + B3*BE31*AL3*AL1 + B3*BE32*AL3*AL2
    #               + B4*BE41*AL4*AL1 + B4*BE42*AL4*AL2 + B4*BE43*AL4*AL3 = 1/30
    # Elements: 8BE1=B2*BE21*AL2*AL1, 8BE2=B3*BE31*AL3*AL1, 8BE3=B3*BE32*AL3*AL2
    #           8BE4=B4*BE41*AL4*AL1, 8BE5=B4*BE42*AL4*AL2, 8BE6=B4*BE43*AL4*AL3
    ExaModels.@add_con(c, x[5]*x[7]*x[6]*x[3] + x[9]*x[11]*x[10]*x[3] + x[9]*x[12]*x[10]*x[6]
                        + x[14]*x[16]*x[15]*x[3] + x[14]*x[17]*x[15]*x[6] + x[14]*x[18]*x[15]*x[10] - T(1.0/30.0))

    # Constraint 8C: B2*AL1^2*BE21 + B3*AL1^2*BE31 + B4*AL1^2*BE41 = 1/40
    # Elements: 8CE1=B2*AL1^2*BE21 (actually XY2Z: X*Y*Y*Z = B2*(AL1^2)*BE21)
    #   wait: EV XY2Z X Y Z, F = X*Y*Y*Z
    #   8CE1: X=B2, Y=AL1, Z=BE21 → B2*AL1*AL1*BE21
    #   8CE2: X=B3, Y=AL1, Z=BE31 → B3*AL1*AL1*BE31
    #   8CE3: X=B4, Y=AL1, Z=BE41 → B4*AL1*AL1*BE41
    ExaModels.@add_con(c, x[5]*x[3]^2*x[7] + x[9]*x[3]^2*x[11] + x[14]*x[3]^2*x[16] - T(1.0/40.0))

    # Constraint 9: B3*BE32*BE21 + B4*BE42*BE21 + B4*(BE43*(BE31+BE32)) = 1/120
    # Elements: 9E1=B3*BE32*BE21 (3PR: X=B3, Y=BE32, Z=BE21)
    #           9E2=B4*BE42*BE21 (3PR: X=B4, Y=BE42, Z=BE21)
    #           9E3=B4*BE43*(BE31+BE32) (3PRI2: X=B4, Y=BE43, Z1=BE31, Z2=BE32; ZZ=Z1+Z2)
    ExaModels.@add_con(c, x[9]*x[12]*x[7] + x[14]*x[17]*x[7] + x[14]*x[18]*(x[11]+x[12]) - T(1.0/120.0))

    # Objective: minimize 0 (NOR2 → least-squares system, ExaModels needs an objective)
    # The SIF has no XN groups — it's a feasibility problem. Use zero objective.
    # Actually NOR2 means we minimize sum of squared residuals in CUTEst terms.
    # But the SIF has all E (equality) groups with no XN — it's NOR2 classification meaning
    # residuals = constraint values. ExaModels will expose these as constraints.
    # We need a dummy objective to have a valid NLPModel.
    ExaModels.@add_obj(c, T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
