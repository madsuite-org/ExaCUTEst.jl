# S316-322: 2-variable QP/NLP with equality constraint (DEN=100 for S316)
# Source: Schittkowski "More Test Examples for NLP Codes", problems 316-322
# Objective:
#   OBJ1: L2 group, v = x1 - 20 -> (x1-20)^2
#   OBJ2: L2 group, v = x2 + 20 -> (x2+20)^2  [note RHS for OBJ2 is -20]
#   Wait: CONSTANTS S316-322 OBJ1 20.0 means RHS=20, so v = x1 - 20
#   CONSTANTS S316-322 OBJ2 -20.0 means RHS=-20, so v = x2 - (-20) = x2+20
# Actually check: group type is linear in GROUPS section:
#   N OBJ1 X1 1.0  (coefficient of X1 is 1)
#   N OBJ2 X2 1.0  (coefficient of X2 is 1)
#   CONSTANTS: OBJ1 RHS=20, OBJ2 RHS=-20
#   So v_OBJ1 = 1*x1 - 20 = x1 - 20, applied L2: (x1-20)^2
#   And v_OBJ2 = 1*x2 - (-20) = x2 + 20, applied L2: (x2+20)^2
# Wait, CONSTANTS section sets -g_rhs. So in SIF:
#   v_g = linear_part - rhs   where CONSTANTS says rhs
#   v_OBJ1 = x1 - 20  -> L2: (x1-20)^2
#   v_OBJ2 = x2 - (-20) = x2 + 20  -> L2: (x2+20)^2  ???
# Hmm but CONSTANTS says X S316-322 OBJ2 -20.0 which means rhs=-20:
#   v_OBJ2 = x2 - (-20) = x2 + 20  -> (x2+20)^2
# Actually check typical convention: CONSTANTS entry sets b such that v = linear - b
# S316-322 OBJ2 -20.0 -> b = -20 -> v = x2 - (-20) = x2 + 20
# This seems unusual. Let me just code it directly.
# Constraint CON (E type, equality):
#   E CON: element X1SQ with coeff 0.01: 0.01*x1^2
#          element X2SQ with coeff 1/DEN: (1/100)*x2^2
#          RHS = 1
#   v_CON = 0.01*x1^2 + (1/DEN)*x2^2 - 1 = 0
# Bounds: FR 'DEFAULT' -> all free
# Start: no start point defined (default 0)
function s316_322_model(DEN = 100.0; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf))
    # Objective: (x1-20)^2 + (x2+20)^2
    ExaModels.@add_obj(c, (x[1] - T(20))^2 + (x[2] + T(20))^2)
    # Constraint: 0.01*x1^2 + (1/DEN)*x2^2 - 1 = 0
    ExaModels.@add_con(c, T(0.01) * x[1]^2 + x[2]^2 / T(DEN) - T(1))
    return ExaModels.ExaModel(c; kwargs...)
end
