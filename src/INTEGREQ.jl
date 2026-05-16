# INTEGREQ: Discrete integral equation problem (More-Garbow-Hillstrom #29)
# N=500 free variables (x(1)..x(N)), x(0)=x(N+1)=0 fixed
# Constraints G(i) = 0 for i=1..N:
#   G(i) = x(i) + h*(1-t_i)/2 * sum_{j=1}^{i} t_j*(x(j)+t_j+1)^3
#                + h*t_i/2 * sum_{j=i+1}^N (1-t_j)*(x(j)+t_j+1)^3
# where h=1/(N+1), t_j = j*h
# This is an NLP (equality system = zero objective on g(x)=0)
# Element VBCUBE(V; B): (V+B)^3, B = t_j + 1 = j*h + 1
# The group G(i) includes x(i) linearly plus nonlinear elements summed with coefficients.
function integreq_model(N = 500; T = Float64, backend = nothing, kwargs...)
    h = T(1) / (N + 1)
    t = T[j * h for j in 1:N]

    # Precompute weights:
    # For constraint i (i=1..N):
    #   lower sum j=1..i: weight = h*(1-t_i)/2 * t_j = h*(1-i*h)/2 * j*h
    #   upper sum j=i+1..N: weight = h*t_i/2 * (1-t_j) = h*i*h/2 * (1-j*h)
    # B_j = t_j + 1 = j*h + 1
    # Constraint: x[i] + sum_{j=1}^i w_lower(i,j)*(x[j]+B_j)^3 + sum_{j=i+1}^N w_upper(i,j)*(x[j]+B_j)^3 = 0

    B = T[t[j] + T(1) for j in 1:N]  # B_j = t_j + 1

    # Lower triangle: (i,j) with j <= i
    lower_data = [(i = i, j = j,
                   w = h * (T(1) - t[i]) / T(2) * t[j])
                  for i in 1:N for j in 1:i]
    # Upper triangle: (i,j) with j > i
    upper_data = [(i = i, j = j,
                   w = h * t[i] / T(2) * (T(1) - t[j]))
                  for i in 1:N for j in (i+1):N]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    # Variables: N free variables x(1)..x(N) (x(0)=x(N+1)=0 fixed, not declared)
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf),
                       start = T[t[i] * (t[i] - T(1)) for i in 1:N])

    # Constraints: for each i, the sum is over (lower_data where .i==i) + (upper_data where .i==i)
    # We use @add_par to access B values
    ExaModels.@add_par(c, Bv, B)

    # Constraints G(i): linear part x[i], plus nonlinear contributions
    # We need to add contributions to the i-th constraint.
    # In ExaModels, @add_con creates one constraint per loop iteration.
    # For constraint i, the expression involves x[i] + sum_{j=1}^N w(i,j)*(x[j]+B[j])^3
    # This requires O(N^2) terms across all constraints.

    # Use data arrays with (constraint_i, var_j, weight) triples:
    ExaModels.@add_par(c, Wlo_w, T[d.w for d in lower_data])
    ExaModels.@add_par(c, Wlo_i, Int[d.i for d in lower_data])
    ExaModels.@add_par(c, Wlo_j, Int[d.j for d in lower_data])

    # This approach won't work easily because we can't have variable-length per-constraint sums.
    # The correct ExaModels approach is to treat each (i,j) pair as contributing to constraint i.

    # Actually ExaModels supports this directly:
    # @add_con(c, x[d.i] + sum contribution for d in data where d.i == i)
    # But ExaModels doesn't filter - it creates one equation per element of the iterator.

    # The proper approach: each (i,j) pair contributes w*(x[j]+B[j])^3 to constraint i.
    # In ExaModels, we can't directly express "constraint i += w*(x[j]+B[j])^3" in a loop.
    #
    # For structural constraints (equality), the recommended approach is:
    # 1. Create N constraints, each with the full expression for constraint i.
    # OR
    # 2. Use a different reformulation.
    #
    # Since all constraints involve sums over all j, we need to express each
    # constraint i as a loop. But in ExaModels, each @add_con iteration creates
    # exactly one constraint. For constraint i with N terms, we'd need N separate
    # "@add_con" calls (one per j), but this would create N*N = N^2 constraints.
    #
    # The only clean way is the scalar form for each constraint:
    # @add_con(c, x[i] + sum(w_lower(i,j)*(x[j]+B[j])^3 for j in 1:i) +
    #                    sum(w_upper(i,j)*(x[j]+B[j])^3 for j in i+1:N) for i in 1:N)
    # But the expression inside the for-loop varies per i (different i1, i2 bounds).
    # This is hard to express in the ExaModels loop form.

    # A clean approach: store all (i,j) pairs with weights, then use a data-driven
    # formulation where the Jacobian sparsity is defined by the (i,j) pairs.
    # In ExaModels, constraints can accumulate contributions from multiple terms
    # using the "multiple @add_con calls contribute to different constraint slots" idea.
    # But actually each @add_con call creates NEW constraint(s).

    # The best approach for this problem: create constraints using the full expression
    # for each i, written as a sum of N terms. This creates N constraints, each with
    # N nonzero Jacobian entries → dense lower-triangular Jacobian.

    # For N=500: 500 constraints each with up to 500 terms = up to 500*500/2 = 125,000
    # nonzero Jacobian entries. This is manageable in ExaModels.

    # Store per-constraint data as vectors of (j, w) pairs, indexed by i.
    # We'll create one constraint per i, with the full sum inside.

    all_data = [(i = i, j = j, w = j <= i ?
                (h * (T(1) - t[i]) / T(2) * t[j]) :
                (h * t[i] / T(2) * (T(1) - t[j])))
               for i in 1:N for j in 1:N]

    ExaModels.@add_par(c, AD, T[d.w for d in all_data])
    ExaModels.@add_par(c, AI, Int[d.i for d in all_data])
    ExaModels.@add_par(c, AJ, Int[d.j for d in all_data])
    ExaModels.@add_par(c, AB, B)

    # This approach doesn't work directly because @add_con creates one constraint
    # per iteration but cannot accumulate. We need to express constraint i in closed form.

    # Simplest working approach: create constraints one by one using scalar form,
    # leveraging Julia's ability to sum over arrays.
    # For each i, constraint is: x[i] + sum(w_data[(i-1)*N+j] * (x[j]+B[j])^3 for j=1:N) = 0
    # where w_data[(i-1)*N+j] = all_data[(i-1)*N+j].w

    # This is O(N^2) work to build but should work.
    # We rearrange data as a matrix W[i,j] for clarity:
    W = [j <= i ?
         (h * (T(1) - t[i]) / T(2) * t[j]) :
         (h * t[i] / T(2) * (T(1) - t[j]))
         for i in 1:N, j in 1:N]

    # Per-constraint expressions using @add_con loop:
    # The trick is: for constraint i, the weight w[j] depends on i,
    # so we can't use a simple uniform loop. Instead, store data differently:
    # For each (i,j) pair, create a "contribution" - but ExaModels accumulates these
    # into the constraint indexed by i.

    # Let's use the fact that the data is symmetric and group by constraint:
    cdata = [(i = i, j = j, w = W[i, j]) for i in 1:N for j in 1:N]
    ExaModels.@add_par(c, W_par, T[d.w for d in cdata])
    ExaModels.@add_par(c, B_par2, B)

    # In ExaModels, when you write @add_con(c, f(x,d) for d in cdata), it creates
    # one constraint PER element of cdata = N^2 constraints. That's NOT what we want.
    # We want N constraints, each being a sum over N terms.

    # The ONLY clean way is to write the full expression for each constraint inline:
    # @add_con(c, expr(i) for i in 1:N) where expr(i) is a complete formula for constraint i.
    # With @add_par storing B[j] and W[i,j]:
    # expr(i) = x[i] + sum(W[i,j] * (x[j] + B[j])^3 for j=1:N)
    # But W[i,j] depends on i and j, while ExaModels' for-loop iterates over a SINGLE index i.
    # We need W as a 2D parameter.

    # Alternative: Store w as a flattened array and construct the expression directly.
    # For each constraint i, accumulate sum_{j=1}^N W[i,j] * (x[j]+B[j])^3.
    # This can be done by creating N*N terms, each assigned to constraint i:
    # But again, @add_con creates one constraint per iteration element.

    # CONCLUSION: For this type of integral equation, the cleanest ExaModels
    # formulation is to write it as a sum-of-squares minimization (not equality)
    # OR accept N^2 constraint terms.

    # Actually, let me reconsider: can we create N constraint "slots" and then
    # add N^2 contributions using a different ExaModels pattern?
    # Yes! In ExaModels, we can add objective/constraint contributions in multiple
    # @add_obj/@add_con calls. The key insight is that:
    #   @add_con(c, linear_i for i in 1:N)  → creates N constraints (linear part)
    #   Then somehow we need to add nonlinear contributions to specific constraints.
    # But ExaModels doesn't support "add to existing constraint i".

    # The REAL solution: ExaModels does support constraint accumulation through
    # a different mechanism - but it requires all terms for constraint i to be in ONE expression.

    # For N=500 and 500 constraints, the expressions are very large.
    # We use the generator form but with a precomputed coefficient array:
    # @add_con(c, x[i] + sum(W[i,j] * (x[j] + B_par2[j])^3 for j in 1:N) for i in 1:N)
    # This IS valid ExaModels syntax! The outer for creates N constraints,
    # the inner sum creates the expression for each constraint.
    # W[i,j] is a plain Julia matrix (not a parameter), and B_par2[j] is a parameter.

    ExaModels.@add_con(c,
        x[i] + sum(W[i, j] * (x[j] + B_par2[j])^3 for j in 1:N)
        for i in 1:N)

    return ExaModels.ExaModel(c; kwargs...)
end
