# KISSING2: Second formulation of kissing number problem
# m=25 points in n=4 dimensional space
# min sum_{i=1}^m <p_i, p_i> - m*n = sum_{i=1}^m ||p_i||^2 - m*n
# s.t. ||p_i - p_j||^2 >= 4 for all i != j
#      ||p_i||^2 >= 4 for all i  (i.e., P(i,i) constraints)
#      Normalization: P(I,J)=0 for 2<=I, I<=J<=n (upper triangle)
# Variables: P(i,j) for i=1..m, j=1..n (m*n = 100 variables)
# With P(i,j)=0 fixed for 2<=i<=n, i<=j<=n (upper triangle, excluding row 1)
# PROD2 element: IV P = Q - R → F = P^2 → (Q-R)^2
# PROD1 element: F = P^2
function kissing2_model(m = 25, n = 4; T = Float64, backend = nothing, kwargs...)
    # Variable index: P(i,j) → (i-1)*n + j
    nvar = m * n

    # Fixed (XX) variables: P(I,J)=0 for 2<=I<=n, I<=J<=n
    # These are variables with XX bounds (fixed to 0)
    fixed = Set{Int}()
    for i_idx in 2:n
        for j_idx in i_idx:n
            fixed_var = (i_idx - 1) * n + j_idx  # variable index within P(i,j) for point i
            # Wait: P(I,J) where I is dimension (2..n) and J is dimension (I..n)?
            # SIF: DO I 2 n-1; DO J I n; XX P(I,J) 0.0 → fix P(I,J)=0 for 2<=I<=n-1, I<=J<=n
            # This means for point index I (which refers to row I of the coordinate matrix),
            # coordinates I..n are fixed to 0.
            # Actually P(I,J) where I is POINT index 2..n and J is DIMENSION index I..n
            # P(I,J) → variable index (I-1)*n + J
            push!(fixed, (i_idx - 1) * n + j_idx)
        end
    end

    # Start: computed from SIF 2*pi*I/m rotation
    start = zeros(T, nvar)
    for i in 1:m
        angle = T(2) * T(pi) * i / m
        c_val = cos(angle)
        s_val = sin(angle)
        # P(i,1) = 2*cos(2*pi*i/m), P(i,j) = 2*sin(2*pi*i/m) for j=2..n-1, P(i,n) = 2*cos(...)
        # From SIF: R+ cos+cos = 2*cos, sin+sin = 2*sin
        start[(i-1)*n + 1] = T(2) * c_val
        for j in 2:(n-1)
            start[(i-1)*n + j] = T(2) * s_val
        end
        start[(i-1)*n + n] = T(2) * c_val
    end

    # Bounds: all free except fixed = 0
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    for idx in fixed
        lvar[idx] = T(0)
        uvar[idx] = T(0)
        start[idx] = T(0)
    end

    # mn = m*n (constant to subtract from objective)
    mn = T(m * n)

    # Objective: sum_i ||p_i||^2 - m*n = sum_{i,k} P(i,k)^2 - mn
    # P(I,J)^2 terms via PROD1 elements
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    ExaModels.@add_obj(c, x[d.idx]^2 for d in [(idx = (i-1)*n + k,) for i in 1:m for k in 1:n])
    # Constant -mn: ExaModels doesn't support constant objective terms directly,
    # but we can add it as a parameter (or omit - CUTEst would match).
    # Actually CUTEst includes the constant in the objective. We need to subtract mn.
    # Use a dummy variable workaround: add -mn as objective constant.
    # In ExaModels there's no direct way to add a scalar constant to the objective.
    # But the GROUPS section says: CONSTANTS: Z KISSING2 OBJ mn → rhs = mn for objective
    # Identity group OBJ with linear part (none) and rhs mn means: group value = -mn
    # contribution = -mn (identity type)
    # So objective = sum_{i,k} x_{i,k}^2 - mn
    # We can model this as: obj = sum x_ij^2 - mn (a constant offset)
    # Since ExaModels can't have a constant, let's add it as a scalar @add_par variable.
    # Actually for CUTEst test compatibility, obj(x0) must match - let's check if
    # ExaModels ignores constant offset... it probably doesn't add it correctly.
    # Looking at other ExaModels implementations: there's no direct constant-in-objective.
    # One workaround: add a "constant" via an extra variable fixed to 1 with linear coefficient.
    # But that changes nvar. Better: just add -mn*1 using a scalar obj contribution.
    # Actually in ExaModels @add_obj(c, -mn) adds the constant -mn to the objective.

    ExaModels.@add_obj(c, -mn)

    # Inequality constraints C(I,J): ||p_I - p_J||^2 >= 4 for all I,J (including I=J)
    # For I != J: PROD2 element with IV P = Q - R where Q=P(I,k), R=P(J,k) → P^2=(P(I,k)-P(J,k))^2
    #   sum_k (P(I,k)-P(J,k))^2 >= 4
    # For I = J: PROD1 element → sum_k P(I,k)^2 >= 4

    # Self-distance (I=J): ||p_i||^2 >= 4
    ExaModels.@add_con(c,
        sum(x[(i-1)*n + k]^2 for k in 1:n) - T(4)
        for i in 1:m;
        lcon = T(0), ucon = T(Inf))

    # Cross-distance (I != J): ||p_I - p_J||^2 >= 4
    cross_pairs = [(ii = ii, jj = jj) for ii in 1:m for jj in 1:m if ii != jj]
    ExaModels.@add_con(c,
        sum((x[(d.ii-1)*n + k] - x[(d.jj-1)*n + k])^2 for k in 1:n) - T(4)
        for d in cross_pairs;
        lcon = T(0), ucon = T(Inf))

    return ExaModels.ExaModel(c; kwargs...)
end
