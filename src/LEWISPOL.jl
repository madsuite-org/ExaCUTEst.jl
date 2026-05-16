# LEWISPOL: Lewis Polynomial Problem - transform of integer programming problem
# 6 variables A(0)..A(5), bounds [-10,10]
# min sum_{j=0}^5 A(j)^2 (SQ elements)
# s.t. D(0): sum_{j=0}^5 A(j) = -1.0 [C(0,j) = 1 for all j]
#     D(1): sum_{j=1}^5 j*A(j) = -6.0 [C(1,j) = j for j=1..5]
#     D(2): sum_{j=2}^5 j*(j-1)*A(j) = -30.0 [C(2,j) = j*(j-1) for j=2..5]
#     INT(j): A(j)^3 - (scale=PEN) → PEN*(A(j)^3 - A(j)^3/...) hmm
# Actually INT(j) groups: XE INT(j) A(j) -1.0, SCALE PEN, element E(j)=CB(A(j))=A(j)^3
# Group value = -A(j) + A(j)^3 = A(j)^3 - A(j) (identity type with scale PEN)
# But scale PEN → contribution = PEN*(v_j/PEN) = v_j (identity with scale doesn't change contribution)
# Actually for identity group type with SCALE sigma: contribution = sigma * (v/sigma) = v
# So INT(j) = PEN*(A(j)^3/PEN - A(j)/PEN) = A(j)^3 - A(j)... no
# Identity group with SCALE sigma: the group value is v, and the function is F(v)=v,
# contribution = sigma * F(v/sigma) = sigma * (v/sigma) = v. So scale doesn't matter for identity.
# Wait, re-reading CLAUDE.md: "Without a GROUP TYPE the function is the identity (F(v)=v),
# contribution = v for objective groups (scale ignored)."
# So INT(j) contribution = -A(j) + A(j)^3 (the group value, scale ignored for identity)
# INT(j) is a constraint (equality? Let me check SIF...)
# SIF: INT(j) group: XE INT(J) A(J) -1.0, SCALE PEN → this is an EQUALITY constraint?
# No - from earlier reading: "DO J 0 N-1; XE INT(J) A(J) -1.0; ZE INT(J) 'SCALE' PEN"
# 'XE' prefix means E type (equality constraint), not N type.
# So INT(j) is an EQUALITY CONSTRAINT: A(j)^3 - A(j) = 0
# But with scale PEN → constraint residual = (A(j)^3 - A(j)) / PEN
# Wait, CLAUDE.md says for identity constraint with SCALE sigma: residual = v/sigma
# So INT(j) residual = (-A(j) + A(j)^3) / PEN = (A(j)^3 - A(j)) / PEN = 0
# → A(j)^3 = A(j) → A(j) ∈ {-1, 0, 1}
# Start: A0=-1, A1=1, A2=1, A3=0, A4=1, A5=-1
# Variable ordering: A(0)..A(5) → x[1]..x[6]
function lewispol_model(; T = Float64, backend = nothing, kwargs...)
    N = 6
    DEG = 3
    PEN = T(1.0e4)

    # Polynomial constraints D(i) for i=0..DEG-1:
    # D(0): sum_{j=0}^5 C(0,j)*A(j) = CT(0) = -1
    #   C(0,j) = 1 for all j
    # D(1): sum_{j=1}^5 C(1,j)*A(j) = CT(1) = -6
    #   C(1,j) = j
    # D(2): sum_{j=2}^5 C(2,j)*A(j) = CT(2) = -30
    #   C(2,j) = j*(j-1)
    # CT(i) = prod_{k=0}^{i-1} C(k, N-k) where C(k,N+1-k)...
    # From SIF: CT(0)=-1, CT(1)=CT(0)*(-1)*(N+1-1)=(-1)*(-1)*6=... actually:
    # CT(0) = AE = -1.0
    # CT(i) = CT(i-1) * (N+1-I) where I goes from 1 to DEG-1=2
    # CT(1) = CT(0) * (N+1-1) = (-1) * 6 = -6
    # CT(2) = CT(1) * (N+1-2) = (-6) * 5 = -30
    ct = [T(-1), T(-6), T(-30)]

    # Coefficient C(i,j) = C(i-1,j) * j (recursion), C(0,j) = 1
    # C(0,j) = 1 for j=0..5
    # C(1,j) = C(0,j) * j = j for j=0..5 (but only j>=1 matter since C(1,0)=0 doesn't help)
    # C(2,j) = C(1,j) * j = j^2 for j=0..5... but C(1,0)=0 so C(2,0)=0, C(2,1)=1, etc.
    # Actually from SIF:
    #   D(0): ZE D0 A(J) C(0,J) = 1.0 for j=0..5
    #   D(1): ZE D(1) A(J) C(1,J) for j=1..5 (inner loop starts at I=1)
    #         C(1,j) = C(0,j) * j = j
    #   D(2): ZE D(2) A(J) C(2,J) for j=2..5
    #         C(2,j) = C(1,j) * j = j^2
    # Wait SIF says:
    #   DO I 1 DEG-1=2
    #   DO J I 5 [not 0..5, starts at I]
    #   C(I,J) = C(I-1,J) * J
    # So: C(1,j) for j=1..5: C(1,j) = C(0,j)*j = j
    # And C(2,j) for j=2..5: C(2,j) = C(1,j)*j = j*j = j^2
    # Note C(2,1) is not computed (loop starts at j=I=2)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-10), uvar = T(10),
                       start = T[-1, 1, 1, 0, 1, -1])

    # Objective: sum_{j=0}^5 A(j)^2 (SQ elements)
    ExaModels.@add_obj(c, x[j]^2 for j in 1:N)

    # D(0): sum_{j=1}^6 x[j] = -1
    ExaModels.@add_con(c, sum(x[j] for j in 1:N) - ct[1])

    # D(1): sum_{j=2}^6 (j-1)*x[j] = -6 [j-1 because A(j-1) is x[j] in 0-indexed]
    # Actually A(J) for J=0..5 → x[1]..x[6], so J=j-1, and C(1,J)=J for J=1..5
    # → sum_{J=1}^5 J * A(J) = -6 → sum_{j=2}^6 (j-1)*x[j] = -6
    ExaModels.@add_con(c, sum(T(j - 1) * x[j] for j in 2:N) - ct[2])

    # D(2): sum_{J=2}^5 J^2 * A(J) = -30 → sum_{j=3}^6 (j-1)^2 * x[j] = -30
    ExaModels.@add_con(c, sum(T(j - 1)^2 * x[j] for j in 3:N) - ct[3])

    # INT(j): (A(j)^3 - A(j)) / PEN = 0 for j=0..5 (equality constraints)
    # With identity group and SCALE=PEN: residual = v/PEN where v = -A(j) + A(j)^3
    # = (A(j)^3 - A(j)) / PEN = 0
    ExaModels.@add_con(c, (x[j]^3 - x[j]) / PEN for j in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
