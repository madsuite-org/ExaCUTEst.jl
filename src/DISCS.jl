# DISCS: Disk packing problem for planar graph representation
# Source: W. Pulleyblank, private communication, 1991
# SIF input: A.R. Conn and Ph. Toint, April 1991
# classification LQR2-MY-36-66
#
# 12 nodes, each with X(I), Y(I), R(I) -> 36 variables
# Variable layout: x[3I-2]=X(I), x[3I-1]=Y(I), x[3I]=R(I), I=1..12
#
# 66 constraints: 18 equality (adjacent) + 48 inequality (non-adjacent)
# S(J,I) = (R(I)+R(J))^2 - (X(I)-X(J))^2 - (Y(I)-Y(J))^2
#   adjacent:     S(J,I) = 0   (touching discs)
#   non-adjacent: S(J,I) + EPSIL <= 0  (disjoint discs)
#
# Constraint ordering determined empirically from CUTEst (not simple SIF traversal):
# - Equality constraints (g[1..18]) in CUTEst order
# - Inequality constraints (g[19..66]) in CUTEst order
#
# Fixed variables (XX): X(1)=0, Y(1)=0, Y(2)=0 (cancel rotation invariance)
# Bounds: R(I) >= 1; all others free
function discs_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))

    N = 12
    EPSIL = T(0.0001)

    # Equality constraint pairs in CUTEst ordering (empirically determined)
    eq_data = [
        (j=1,i=2),(j=11,i=12),(j=2,i=3),(j=10,i=12),(j=2,i=4),(j=3,i=4),
        (j=10,i=11),(j=5,i=11),(j=9,i=10),(j=4,i=5),(j=8,i=10),(j=8,i=9),
        (j=7,i=9),(j=4,i=6),(j=5,i=6),(j=1,i=7),(j=7,i=8),(j=6,i=7)
    ]

    # Inequality constraint pairs in CUTEst ordering (empirically determined)
    iq_data = [
        (j=4,i=7),(j=5,i=7),(j=3,i=7),(j=1,i=8),(j=2,i=8),(j=3,i=8),(j=4,i=8),(j=5,i=8),(j=6,i=8),
        (j=2,i=7),(j=1,i=9),(j=2,i=9),(j=3,i=9),(j=4,i=9),(j=5,i=9),(j=6,i=9),
        (j=3,i=6),(j=2,i=6),(j=1,i=10),(j=2,i=10),(j=3,i=10),(j=4,i=10),(j=5,i=10),(j=6,i=10),(j=7,i=10),
        (j=1,i=6),(j=3,i=5),(j=1,i=11),(j=2,i=11),(j=3,i=11),(j=4,i=11),
        (j=2,i=5),(j=6,i=11),(j=7,i=11),(j=8,i=11),(j=9,i=11),
        (j=1,i=5),(j=1,i=12),(j=2,i=12),(j=3,i=12),(j=4,i=12),(j=5,i=12),(j=6,i=12),(j=7,i=12),(j=8,i=12),(j=9,i=12),
        (j=1,i=4),(j=1,i=3)
    ]

    # Variable layout: x[3I-2]=X(I), x[3I-1]=Y(I), x[3I]=R(I)
    lvar = fill(T(-Inf), 3*N)
    uvar = fill(T(Inf), 3*N)
    x0   = zeros(T, 3*N)
    for i in 1:N
        lvar[3i]   = T(1)              # R(I) >= 1
        x0[3i-2]   = T(0.03*i)        # X(I) start
        x0[3i-1]   = T(0.0055555*i)   # Y(I) start
        x0[3i]     = T(float(i))       # R(I) start
    end
    # Fixed: X(1)=0, Y(1)=0, Y(2)=0 (bounds fixed; start values from SIF are kept)
    # Note: CUTEst reports x0 = start point values even for fixed variables
    lvar[1] = T(0); uvar[1] = T(0)   # X(1) fixed at 0
    lvar[2] = T(0); uvar[2] = T(0)   # Y(1) fixed at 0
    lvar[5] = T(0); uvar[5] = T(0)   # Y(2) fixed at 0

    ExaModels.@add_var(c, x, 3*N; lvar = lvar, uvar = uvar, start = x0)

    # Objective: minimize sum of R(I) = sum x[3i]
    ExaModels.@add_obj(c, x[3*i] for i in 1:N)

    # Equality constraints (adjacent pairs): S(J,I) = 0
    ExaModels.@add_con(c,
        (x[3*d.i] + x[3*d.j])^2 -
        (x[3*d.i-2] - x[3*d.j-2])^2 -
        (x[3*d.i-1] - x[3*d.j-1])^2
        for d in eq_data)

    # Inequality constraints (non-adjacent pairs): S(J,I) + EPSIL <= 0
    # CONSTANTS: Z DISCS S(J,I) -EPSIL → group value v ≤ -EPSIL → residual = v-(-EPSIL) ≤ 0
    ExaModels.@add_con(c,
        (x[3*d.i] + x[3*d.j])^2 -
        (x[3*d.i-2] - x[3*d.j-2])^2 -
        (x[3*d.i-1] - x[3*d.j-1])^2 + EPSIL
        for d in iq_data;
        lcon = T(-Inf), ucon = T(0))

    return ExaModels.ExaModel(c; kwargs...)
end
