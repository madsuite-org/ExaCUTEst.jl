# MSS1: Maximum stable set rank-2 relaxation (45 vertices, 72 edges)
# SIF: N. Gould, March 2002.  classification QQR2-AN-90-73
#
# Variables: X(K), Y(K) for K=1:N; free, starting at 1.0
# Ordering: X(1), Y(1), X(2), Y(2), ..., X(N), Y(N)
#
# Objective: -( sum X(K) )^2 - ( sum Y(K) )^2
#   via -L2 group type applied to groups O1 (sum X(K)) and O2 (sum Y(K))
#
# Constraints:
#   S: sum X(K)^2 + Y(K)^2 = 1  (spherical constraint, 1 equality)
#   C(K): X(a_k)*X(b_k) + Y(a_k)*Y(b_k) = 0  for each edge (a_k,b_k), K=1:E
#
# Constraint order: S first, then C(1..E)

const _MSS1_N = 45
const _MSS1_E = 72
const _MSS1_EDGES = Tuple{Int,Int}[
    (10,1), (11,2), (11,10), (12,3), (12,10), (12,11), (13,4), (14,5),
    (14,13), (15,6), (15,13), (15,14), (16,7), (17,8), (17,16), (18,9),
    (18,16), (18,17), (19,1), (20,5), (20,19), (21,9), (21,19), (21,20),
    (22,1), (23,4), (23,22), (24,8), (24,22), (24,23), (25,1), (26,6),
    (26,25), (27,7), (27,25), (27,26), (28,2), (29,4), (29,28), (30,9),
    (30,28), (30,29), (31,2), (32,5), (32,31), (33,7), (33,31), (33,32),
    (34,2), (35,6), (35,34), (36,8), (36,34), (36,35), (37,3), (38,6),
    (38,37), (39,9), (39,37), (39,38), (40,3), (41,5), (41,40), (42,8),
    (42,40), (42,41), (43,3), (44,4), (44,43), (45,7), (45,43), (45,44),
]

function mss1_model(; T = Float64, backend = nothing, kwargs...)
    N = _MSS1_N
    E = _MSS1_E
    edges = _MSS1_EDGES

    nvar = 2 * N
    Xi(k) = 2 * (k - 1) + 1   # X(k) index
    Yi(k) = 2 * (k - 1) + 2   # Y(k) index

    lvar  = fill(T(-Inf), nvar)
    uvar  = fill(T(Inf),  nvar)
    start = fill(T(1.0),  nvar)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nvar; lvar = lvar, uvar = uvar, start = start)

    # ── Objective: -(sum X)^2 - (sum Y)^2 ────────────────────────────────────
    xidx = [Xi(k) for k in 1:N]
    yidx = [Yi(k) for k in 1:N]
    ExaModels.@add_obj(c, -(sum(x[xidx[k]] for k in 1:N))^2)
    ExaModels.@add_obj(c, -(sum(x[yidx[k]] for k in 1:N))^2)

    # ── Spherical constraint: sum X(K)^2 + Y(K)^2 - 1 = 0 ───────────────────
    ExaModels.@add_con(c,
        sum(x[Xi(k)]^2 + x[Yi(k)]^2 for k in 1:N) - T(1))

    # ── Edge constraints: X(a)*X(b) + Y(a)*Y(b) = 0 for each edge ───────────
    edge_data = [(a = Xi(e[1]), b = Xi(e[2]), c_ = Yi(e[1]), d_ = Yi(e[2]))
                 for e in edges]
    ExaModels.@add_con(c,
        x[d.a] * x[d.b] + x[d.c_] * x[d.d_]
        for d in edge_data)

    return ExaModels.ExaModel(c; kwargs...)
end
