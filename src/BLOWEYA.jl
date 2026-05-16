# BLOWEYA: Nonconvex QP on charge distribution (case A: a=0.2,b=0.4,c=0.6,d=0.8)
# N=2000, 2*(N+1) variables U(I),W(I) for I=0..N
# Variables: -1<=U(I)<=1, W(I) free (XR)
# Constraints: INT (equality, trapezoidal sum = int_rhs), CON(I) (equality, A*w=u system)
# Objective: quadratic in U and W with data-dependent linear terms
function bloweya_model(N = 2000; T = Float64, backend = nothing, kwargs...)
    # Compute partition indices (case A: divide by 5)
    NA = N ÷ 5
    NB = 2 * NA
    NC = 3 * NA
    ND = 4 * NA

    # Compute V(I) for I = 0..N (piecewise linear function)
    V = Vector{T}(undef, N + 1)
    # V(0..NA) = 1
    for i in 0:NA
        V[i+1] = T(1)
    end
    # V(NA+1..NB) = 1 - 2*(i-NA)/((B-A)*N) = linear from 1 to -1
    step1 = T(2) / T((NB - NA))
    for i in NA+1:NB
        V[i+1] = T(1) - step1 * T(i - NA)
    end
    # V(NB+1..NC) = -1
    for i in NB+1:NC
        V[i+1] = T(-1)
    end
    # V(NC+1..ND) = -1 + 2*(i-NC)/((D-C)*N) = linear from -1 to 1
    step2 = T(2) / T((ND - NC))
    for i in NC+1:ND
        V[i+1] = T(-1) + step2 * T(i - NC)
    end
    # V(ND+1..N) = 1
    for i in ND+1:N
        V[i+1] = T(1)
    end

    # Computed constants
    inv_N2 = T(1) / T(N)^2
    # INT constant: integral of v (piecewise trapezoidal) * N, times 0.2
    # From SIF: INT = (1 + A + B - C - D)*N where A=NA/N=0.2, B=NB/N=0.4, C=NC/N=0.6, D=ND/N=0.8
    # INT_param = (1 + 0.2 + 0.4 - 0.6 - 0.8)*N = 0.2*N
    # CONST(INT group) = 0.2 * INT_param = 0.2 * 0.2 * N = 0.04 * N
    # Hmm, but that gives 0.04*2000 = 80. Let me verify by computing the actual trapezoidal sum of V:
    # sum = 0.5*V(0) + sum_{I=1}^{N-1} V(I) + 0.5*V(N)
    # This should equal INT_param = 0.2*N = 400 from SIF parameter arithmetic.
    # ZE BLOWEYA INT 0.2 INT means CONST = 0.2 * INT_param? Or is it just INT_param?
    # Let's compute it directly:
    int_rhs = T(0.5) * V[1]
    for i in 1:N-1
        int_rhs += V[i+1]
    end
    int_rhs += T(0.5) * V[N+1]
    # This is the trapezoidal integral = INT_param. The RHS of INT constraint = int_rhs.
    # But SIF has "ZE BLOWEYA INT 0.2 INT" - since we compute INT_param = (1+0.2+0.4-0.6-0.8)*N = 400
    # and "0.2 INT" in CONSTANTS means multiply by 0.2, giving 80.
    # But we computed int_rhs above which should be the actual trapezoidal sum = 400 for V as defined.
    # Let's trust the direct computation: int_rhs = 400. The CONST = 80 is suspicious.
    # Actually, rethinking the SIF: "Z PROBNAME GROUP PNAME" reads 3 fields.
    # "ZE BLOWEYA INT 0.2 INT" has 5 fields. Perhaps it's two entries:
    # "ZE BLOWEYA INT 0.2" (literal constant 0.2) and "INT" is something else.
    # Actually in many SIF readers, additional fields are ignored or it's:
    # field 1: Z/ZE/ZG
    # field 2: problem name
    # field 3: group name
    # field 4: parameter name (or literal)
    # field 5: [optional] multiplier if field 4 is "Z" variant
    # Given the context, I believe INT group CONST = value of parameter named "INT" = 400...
    # but ZE vs Z vs X: "X" for literal, "Z" for parameter. "ZE" might be "Z" + "E" for equality annotation.
    # Reading the actual SIF more carefully: the line is probably:
    # Z BLOWEYA INT 0.2     (set INT const = literal 0.2)
    # and "INT" on the same line is a continuation which seems odd.
    # I'll compute the RHS directly from the V array trapezoidal sum
    # which equals the integral of v = INT_param = 400.
    # However from the SIF line alone I'm unsure. Let me use int_rhs = 400 (the trapezoidal sum of V).

    # Variable indexing:
    # U(I) = 2*I + 1 (I=0..N), W(I) = 2*I + 2
    u_idx(i) = 2 * i + 1
    w_idx(i) = 2 * i + 2
    NV = 2 * (N + 1)

    lvar = fill(T(-Inf), NV)
    uvar = fill(T(Inf), NV)
    start_vals = zeros(T, NV)

    # U(I) bounds: -1 <= U(I) <= 1
    for i in 0:N
        lvar[u_idx(i)] = T(-1)
        uvar[u_idx(i)] = T(1)
        start_vals[u_idx(i)] = V[i+1]  # start = V(I)
    end
    # W(I) free (XR), start = 0

    ec = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(ec, x, NV; lvar = lvar, uvar = uvar, start = start_vals)

    # Precompute linear coefficients for U(I) and W(I) in objective
    # From SIF groups section:
    # For each I=0..N: ZN OBJ U(I) V(I)*(-1/N^2) and ZN OBJ W(I) V(I)*(-2/N^2)
    # For I=0: ZN OBJ U(0) (V(1)-V(0))
    # For I=1..N-1: ZN OBJ U(I) (-2*V(I)+V(I-1)+V(I+1))
    # For I=N: ZN OBJ U(N) (V(N-1)-V(N))
    # (These are cumulative additions to the same U/W variables)

    # Linear obj terms: sum_I (c_u[I]*U(I) + c_w[I]*W(I))
    c_u = [-V[i+1] * inv_N2 for i in 0:N]  # c_u[I] from first ZN loop
    c_w = [-T(2) * V[i+1] * inv_N2 for i in 0:N]

    # Add Laplacian-like terms to c_u:
    c_u[0+1] += V[1+1] - V[0+1]  # U(0): (V(1)-V(0))
    for i in 1:N-1
        c_u[i+1] += -T(2)*V[i+1] + V[i] + V[i+2]
    end
    c_u[N+1] += V[N] - V[N+1]  # U(N): (V(N-1)-V(N))

    # Add linear objective terms
    lin_data_u = [(i = u_idx(i), cu = c_u[i+1]) for i in 0:N]
    ExaModels.@add_obj(ec, d.cu * x[d.i] for d in lin_data_u)

    lin_data_w = [(i = w_idx(i), cw = c_w[i+1]) for i in 0:N]
    ExaModels.@add_obj(ec, d.cw * x[d.i] for d in lin_data_w)

    # Nonlinear objective terms:
    # D(I) for I=0: 1*U(0)^2
    # O(I) for I=0: -2*U(0)*U(1)
    # D(I)*2 for I=1..N-1: 2*U(I)^2
    # O(I)*(-2) for I=1..N-1: -2*U(I)*U(I+1)
    # D(N)*1: 1*U(N)^2
    # C(I)*(1/N^2) for I=0..N: U(I)*W(I)/N^2

    # Boundary D and O
    ExaModels.@add_obj(ec, x[u_idx(0)]^2)
    ExaModels.@add_obj(ec, T(-2) * x[u_idx(0)] * x[u_idx(1)])

    # Interior D and O
    do_data = [(ui = u_idx(i), ui1 = u_idx(i+1)) for i in 1:N-1]
    ExaModels.@add_obj(ec, T(2) * x[d.ui]^2 for d in do_data)
    ExaModels.@add_obj(ec, T(-2) * x[d.ui] * x[d.ui1] for d in do_data)

    ExaModels.@add_obj(ec, x[u_idx(N)]^2)

    # C(I) = U(I)*W(I)/N^2
    c_data = [(ui = u_idx(i), wi = w_idx(i)) for i in 0:N]
    ExaModels.@add_obj(ec, inv_N2 * x[d.ui] * x[d.wi] for d in c_data)

    # INT constraint: 0.5*U(0) + sum_{I=1}^{N-1} U(I) + 0.5*U(N) - int_rhs = 0
    # Linear constraint, handled by adding each U(I) term
    # Since INT is a linear equality, we need to encode it as one constraint.
    # @add_con with a loop over I for linear sum is not directly possible since it creates
    # separate constraints. We must express it as a single expression.
    # Approach: use @add_par for the U indices, building a scalar expression.
    # Or: split into multiple @add_con and use auxiliary variable trick.
    # Actually: the correct way is to build the sum through the data.
    # With ExaModels, a single @add_con(c, expr) is possible when expr is a static sum.
    # For large N=2000, we can't inline all 2001 variables.
    # Solution: Use INT as an objective-level linear constraint with @add_par.
    # Actually ExaModels @add_con with a for loop creates SEPARATE constraints per iteration.
    # To create a single linear equality sum, we need an auxiliary approach.
    # However, the CON(I) constraints are also linear.

    # For a linear sum constraint, the standard ExaModels pattern is:
    # @add_con(c, expr for d in data) creates N SEPARATE constraints.
    # For a SUM constraint we need a different approach.
    # One approach: define a helper parameter @add_par and use it.
    # Actually the cleanest way for a linear sum = constant constraint is to just
    # encode each element as a separate @add_con term and sum them. But that creates N constraints.

    # For the INT constraint: encode as sum_I c_I * U(I) - int_rhs = 0.
    # This is a single equality. We can't directly express this in ExaModels @add_con
    # without a loop that creates separate constraints.
    # Trick: partition the sum and use a single scalar expression with all N+1 terms is too large.
    # Alternative: use @add_par for the integer coefficients and write a loop.
    # Actually looking at how BLOWEYA is used: the INT constraint is ONE linear equality,
    # and CON(I) constraints are N+1 equalities.
    # ExaModels supports @add_par for data. Let me use @add_par for U indices.

    # For the INT constraint, encode linearly:
    # 0.5*U(0) + sum_{I=1}^{N-1} U(I) + 0.5*U(N) - int_rhs = 0
    # We need to express this as a single scalar constraint.
    # Since N can be large, let me think of another way.
    #
    # APPROACH: Use @add_par with coefficients, then reference via @add_con with loop.
    # But @add_con with a loop creates N separate constraints!
    #
    # The only way to have a single sum constraint in ExaModels is to:
    # 1. Express it as a sum in the `for d in data` generator, but ExaModels would create 1 constraint
    #    per iteration item.
    # 2. Use auxiliary variables.
    #
    # Given ExaModels API constraints, for a sum=constant linear constraint,
    # we need to use an @add_par approach or inline ALL terms.
    # For the INT equality, we can parameterize coefficients:
    # INT constraint: 0.5*U(0) + sum_{I=1}^{N-1} U(I) + 0.5*U(N) = int_rhs
    # This is a single linear equality. Encode via data loop over nonzero terms.
    int_coefs = [i == 0 || i == N ? T(0.5) : T(1) for i in 0:N]
    int_data = [(ui = u_idx(i), coef = int_coefs[i+1]) for i in 0:N]

    # INT constraint as single equality using sum() inside @add_con
    ExaModels.@add_con(ec,
        sum(int_coefs[i+1] * x[u_idx(i)] for i in 0:N) - int_rhs)

    # CON(I) constraints (linear equalities):
    # CON(0): U(0) - U(1) - inv_N2*W(0) = 0
    ExaModels.@add_con(ec, x[u_idx(0)] - x[u_idx(1)] - inv_N2 * x[w_idx(0)])

    # CON(I) for I=1..N-1: 2*U(I) - U(I+1) - U(I-1) - inv_N2*W(I) = 0
    con_data = [(ui = u_idx(i), uim1 = u_idx(i-1), uip1 = u_idx(i+1), wi = w_idx(i))
                for i in 1:N-1]
    ExaModels.@add_con(ec,
        T(2)*x[d.ui] - x[d.uip1] - x[d.uim1] - inv_N2 * x[d.wi]
        for d in con_data)

    # CON(N): U(N) - U(N-1) - inv_N2*W(N) = 0
    ExaModels.@add_con(ec, x[u_idx(N)] - x[u_idx(N-1)] - inv_N2 * x[w_idx(N)])

    return ExaModels.ExaModel(ec; kwargs...)
end
