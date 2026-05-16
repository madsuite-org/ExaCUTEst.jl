# DIPIGRI: Di Pillo and Grippo problem (7 variables, 4 inequality constraints)
# minimize -(x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2 + 10*x5^6 + 7*x6^2 + x7^4 - 4*x6*x7 - 10*x6 - 8*x7
# Wait - let me re-read the SIF:
# OBJ: linear part: -10*x6 - 8*x7
#      elements: O1=(x1-10)^2, O2=5*(x2-12)^2, O3=x3^4, O4=3*(x4-11)^2, O5=10*x5^6, O6=7*x6^2+x7^4-4*x6*x7
# GROUP USES: OBJ = O1 + 5*O2 + O3 + 3*O4 + 10*O5 + O6
# Wait: E OBJ O1 (coeff=1) O2 5.0 O3 (coeff=1) O4 3.0 O5 10.0 O6 (coeff=1)
# OBJ = (x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2 + 10*x5^6 + 7*x6^2+x7^4-4*x6*x7 - 10*x6 - 8*x7
#
# Constraints (L type, <= 0):
# C1: 2*(x1)^2 + 3*(x2)^4 + 4*(x3)^2 + x3 + 5*x5 - 127 <= 0
#   linear: x3 + 5*x5, elements: C11=2*x1^2, C12=3*x2^4, C13=4*x4^2 -- wait, C13 uses X4
#   Let me re-read: C1 elements: C11=SQ(X1) coeff 2.0, C12=P4(X2) coeff 3.0, C13=SQ(X4) coeff 4.0
#   Linear C1: x3 + 5*x5; CONSTANTS C1 = 127
#   So C1: 2*x1^2 + 3*x2^4 + 4*x4^2 + x3 + 5*x5 - 127 <= 0  (L type)
# C2: 7*x1 + 3*x2 + x4 - x5 + 10*x3^2 - 282 <= 0
#   linear: 7*x1 + 3*x2 + x4 - x5, C21=SQ(X3) coeff 10.0, rhs=282
# C3: 23*x1 + x2^2 + 6*x6^2 - 8*x7 - 196 <= 0
#   linear: 23*x1 - 8*x7, C31=SQ(X2), C32=SQ(X6) coeff 6, rhs=196
# C4: 4*x1^2*x2^2 - 3*x1*x2 + 2*x1^2 - ... wait
#   C4: CE element = 4*x1^2+x2^2-3*x1*x2, plus 2*x3^2*coeff, linear 5*x6-11*x7
#   C41=CE(X1,X2): F = 4*X^2+Y^2-3*X*Y; C21=SQ(X3) coeff 2; linear: 5*x6-11*x7
#   rhs not specified → 0
# All variables free; start given above
function dipigri_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 7; lvar = T(-Inf), uvar = T(Inf),
                       start = T[1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0])
    # Objective:
    # (x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2 + 10*x5^6
    # + 7*x6^2 + x7^4 - 4*x6*x7 - 10*x6 - 8*x7
    ExaModels.@add_obj(c,
        (x[1] - T(10))^2 + T(5) * (x[2] - T(12))^2 + x[3]^4 +
        T(3) * (x[4] - T(11))^2 + T(10) * x[5]^6 +
        T(7) * x[6]^2 + x[7]^4 - T(4) * x[6] * x[7] - T(10) * x[6] - T(8) * x[7])
    # C1 (L <= 0): 2*x1^2 + 3*x2^4 + 4*x4^2 + x3 + 5*x5 - 127 <= 0
    ExaModels.@add_con(c,
        T(2) * x[1]^2 + T(3) * x[2]^4 + T(4) * x[4]^2 + x[3] + T(5) * x[5] - T(127);
        lcon = T(-Inf), ucon = T(0))
    # C2 (L <= 0): 10*x3^2 + 7*x1 + 3*x2 + x4 - x5 - 282 <= 0
    ExaModels.@add_con(c,
        T(10) * x[3]^2 + T(7) * x[1] + T(3) * x[2] + x[4] - x[5] - T(282);
        lcon = T(-Inf), ucon = T(0))
    # C3 (L <= 0): x2^2 + 6*x6^2 + 23*x1 - 8*x7 - 196 <= 0
    ExaModels.@add_con(c,
        x[2]^2 + T(6) * x[6]^2 + T(23) * x[1] - T(8) * x[7] - T(196);
        lcon = T(-Inf), ucon = T(0))
    # C4 (L <= 0): 4*x1^2+x2^2-3*x1*x2 + 2*x3^2 + 5*x6 - 11*x7 <= 0
    ExaModels.@add_con(c,
        T(4) * x[1]^2 + x[2]^2 - T(3) * x[1] * x[2] + T(2) * x[3]^2 + T(5) * x[6] - T(11) * x[7];
        lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
