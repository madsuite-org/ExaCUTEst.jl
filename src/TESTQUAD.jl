# TESTQUAD: Diagonal quadratic with controlled spectrum (N variables, unconstrained)
# minimize sum_i 0.5 * lambda_i * x_i^2
# All variables free, starting point x = 1.0
# Default N = 5000 (matching SIF default IE N = 5000)
#
# SIF structure:
#   One group OBJ (N type), elements ETYPE with F = 0.5*P*V^2
#   Eigenvalues lambda_i computed in four bands:
#     Lower cluster  [L1..U1] = [1..N/4]:         uniform in [1, UPPERL=10]
#     Middle cluster [L2..U2] = [N/4+1..N/2]:     uniform in [LOWERM, UPPERM]
#     Upper cluster  [L3..U3] = [N/2+1..3N/4]:    uniform in [LOWERU, COND]
#     Non-clustered  [L4..U4] = [3N/4+1..N]:      uniform in [1, COND]
function testquad_model(N = 10; T = Float64, backend = nothing, kwargs...)
    # SIF parameter setup
    CLUSTERL = div(N, 4)
    CLUSTERM = div(N, 4)
    CLUSTERU = div(N, 4)

    L1 = 1;          U1 = CLUSTERL
    L2 = U1 + 1;     U2 = U1 + CLUSTERM
    L3 = U2 + 1;     U3 = U2 + CLUSTERU
    L4 = U3 + 1;     U4 = N

    UPPERL  = T(10.0)
    LOWERM  = T(499995.0)
    UPPERM  = T(500005.0)
    LOWERU  = T(999990.0)
    COND    = T(1000000.0)
    ONE     = T(1.0)

    SIZEL   = UPPERL - ONE                   # 9.0
    SIZEM   = UPPERM - LOWERM                # 10.0
    SIZEU   = COND   - LOWERU                # 10.0
    COND1   = COND   - ONE                   # 999999.0

    # Build eigenvalue array
    lambda = Vector{T}(undef, N)

    # Lower cluster: i = L1..U1 (indices 1..CLUSTERL)
    if L1 <= U1
        lambda[L1] = ONE
        if CLUSTERL > 1
            step_l = SIZEL / T(CLUSTERL - 1)
            for i in L1+1:U1
                j = i - L1
                lambda[i] = ONE + T(j) * step_l
            end
        end
    end

    # Middle cluster: i = L2..U2
    if L2 <= U2
        lambda[L2] = LOWERM
        if CLUSTERM > 1
            step_m = SIZEM / T(CLUSTERM - 1)
            for i in L2+1:U2
                j = i - L2
                lambda[i] = LOWERM + T(j) * step_m
            end
        end
    end

    # Upper cluster: i = L3..U3
    if L3 <= U3
        lambda[L3] = LOWERU
        if CLUSTERU > 1
            step_u = SIZEU / T(CLUSTERU - 1)
            for i in L3+1:U3
                j = i - L3
                lambda[i] = LOWERU + T(j) * step_u
            end
        end
    end

    # Non-clustered: i = L4..U4
    REST = U4 - L4  # = N - (3N/4) = N/4  (number of remaining - 1)
    if L4 <= U4
        lambda[L4] = ONE
        if REST > 0
            step_r = COND1 / T(REST)
            for i in L4+1:U4
                j = i - L4
                lambda[i] = ONE + T(j) * step_r
            end
        end
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = T(1.0))
    ExaModels.@add_par(c, λ, lambda)
    ExaModels.@add_obj(c, T(0.5) * λ[i] * x[i]^2 for i in 1:N)
    return ExaModels.ExaModel(c; kwargs...)
end
