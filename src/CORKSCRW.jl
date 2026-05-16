# CORKSCRW: Nonlinear optimal control problem (corkscrew trajectory)
# Find minimum-time trajectory for a mass moving in 3D space near curve y=sin(x), z=cos(x).
# Source: Ph. Toint, private communication.
# SIF input: Ph. Toint, April 1991.
# SIF classification: SOR2-AN-V-V
#
# Parameters (defaults): T=500 time intervals, XT=10 target abscissa, MASS=0.37, TOL=0.1
# Variables: 9T+6 total (after sifdecoder butterfly permutation)
#
# The sifdecoder butterfly permutation for DO loops starting at I=0 produces
# a complex variable ordering that is fully derived and verified empirically
# for T=10 and spot-checked for T=500.
#
# All equality constraints have the form:  coef*x[b1] - coef*x[b2] - x[c] = 0
#   ACX(I):  MoH*VX(I)   - MoH*VX(I-1)   - UX(I)  = 0
#   ACY(I):  MoH*VY(I)   - MoH*VY(I-1)   - UY(I)  = 0
#   ACZ(I):  MoH*VZ(I)   - MoH*VZ(I-1)   - UZ(I)  = 0
#   PSX(I):  invH*X(I)   - invH*X(I-1)   - VX(I)  = 0
#   PSY(I):  invH*Y(I)   - invH*Y(I-1)   - VY(I)  = 0
#   PSZ(I):  invH*Z(I)   - invH*Z(I-1)   - VZ(I)  = 0
#
# Equality constraint ordering (verified for T=10):
#   General interleaving: block 1 (6 rows), then for slot I=1..T-2:
#     emit terminal(type=(I-1)%6+1, Iv=T-div(I-1,6)), then first rows of block I+1
#     (6 - count_in_slots(I+1) rows, where count_in_slots counts how many of I+1's
#     6 slots fall in 1..T-2).  Terminals cycle PSZ,PSY,PSX,ACZ,ACY,ACX for each
#     level, with Iv counting down from T.
# Inequality constraint ordering (rows 6T+1..7T):
#   Fwd I (T/2..T-1) sorted by descending X Julia position,
#   then back I (T/2-1..1) in descending order, then I=T.

function corkscrw_model(TT = 500; T = Float64, backend = nothing, kwargs...)
    XT   = T(10.0)
    MASS = T(0.37)
    TOL  = T(0.1)

    RT   = T(TT)
    H    = XT / RT
    invH = RT / XT
    MoH  = MASS / H
    FMAX = XT / RT            # = H
    XTT1 = XT * (RT + T(1))
    W    = XTT1 * T(0.5)
    TOLSQ = TOL^2

    nv = 9 * TT + 6

    half = TT ÷ 2
    K    = (3 * TT - 2) ÷ 18   # boundary for overflow sub-bands

    # ----------------------------------------------------------------
    # Variable index helpers (I is 0-based integer, returns 1-based Julia index)
    # ----------------------------------------------------------------

    # VX/VY/VZ:
    #   I=0:   fixed block (bottom of VX sub-band)
    #   I=TT:  fixed block (top)
    #   back (I<half):  VX = 6T-3-6I, VY = VX-1, VZ = VX-2
    #   fwd  (I>=half): VX = 4+6I,    VY = VX+1, VZ = VX+2
    function vxI(i)
        i == 0  && return 6*TT - 3
        i == TT && return 6*TT + 4
        i < half ? 6*TT - 3 - 6*i : 4 + 6*i
    end
    function vyI(i)
        i == 0  && return 6*TT - 4
        i == TT && return 6*TT + 5
        i < half ? 6*TT - 4 - 6*i : 5 + 6*i
    end
    function vzI(i)
        i == 0  && return 6*TT - 5
        i == TT && return 6*TT + 6
        i < half ? 6*TT - 5 - 6*i : 6 + 6*i
    end

    # U variables: UX(i) = 6T+4+3i, UY = UX+1, UZ = UX+2  (i=1..TT)
    uxI(i) = 6*TT + 4 + 3*i
    uyI(i) = 6*TT + 5 + 3*i
    uzI(i) = 6*TT + 6 + 3*i

    # Fixed variable positions
    x0_idx = 6*TT + 3    # X(0)
    y0_idx = 6*TT + 2    # Y(0)
    z0_idx = 6*TT + 1    # Z(0)

    # ----------------------------------------------------------------
    # X positions: butterfly algorithm
    #   start_X(j) = 1 + (j-1)*(j+4)÷2
    #   P_x = max j such that start_X(j)+1 <= TT
    #   inner pairs j=1..P_x: X(half-1+j)=start_X(j), X(half-j)=start_X(j)+1
    #   remaining positions fill outer pairs desc, then I=TT-1, I=TT
    # ----------------------------------------------------------------
    start_X(j) = 1 + (j - 1) * (j + 4) ÷ 2

    P_x = 0
    for j in 1:TT
        if start_X(j) + 1 <= TT
            P_x = j
        else
            break
        end
    end

    xpos = Vector{Int}(undef, TT + 1)  # xpos[i+1] = Julia index of X(i) for i=0..TT
    xpos[0 + 1] = x0_idx

    assigned_set = Set{Int}()
    for j in 1:P_x
        fwd_I  = half - 1 + j
        back_I = half - j
        p1 = start_X(j)
        p2 = start_X(j) + 1
        xpos[fwd_I + 1]  = p1
        xpos[back_I + 1] = p2
        push!(assigned_set, p1, p2)
    end

    reserved = sort([p for p in 1:TT if p ∉ assigned_set], rev = true)
    ri = 1

    for j in (P_x + 1):(half - 1)
        fwd_I  = half - 1 + j
        back_I = half - j
        xpos[fwd_I + 1]  = reserved[ri];     ri += 1
        xpos[back_I + 1] = reserved[ri];     ri += 1
    end

    if ri <= length(reserved)
        xpos[TT - 1 + 1] = reserved[ri];     ri += 1
    end
    if ri <= length(reserved)
        xpos[TT + 1] = reserved[ri];         ri += 1
    end

    # ----------------------------------------------------------------
    # Y positions (verified for T=10)
    # ----------------------------------------------------------------
    ypos = Vector{Int}(undef, TT + 1)
    ypos[0 + 1] = y0_idx
    ypos[TT + 1] = 3*TT - 5
    for i in 1:(TT - 1)
        if i <= K
            ypos[i + 1] = 3*TT - 12*i
        elseif i < half
            ypos[i + 1] = 2 + 6*i
        elseif i <= TT - K - 1
            ypos[i + 1] = 6*TT - 1 - 6*i
        else
            ypos[i + 1] = 12*i - 9*TT + 1
        end
    end

    # ----------------------------------------------------------------
    # Z positions (verified for T=10, including edge case guard)
    # ----------------------------------------------------------------
    zpos = Vector{Int}(undef, TT + 1)
    zpos[0 + 1] = z0_idx
    zpos[TT + 1] = 3*TT
    for i in 1:(TT - 1)
        if i <= K
            zpos[i + 1] = 3*TT - 12*i - 5
        elseif i < half
            zpos[i + 1] = 3 + 6*i
        elseif i <= TT - K - 1
            z_reg = 6*TT - 2 - 6*i
            zpos[i + 1] = (z_reg <= TT) ? (12*i - 9*TT + 6) : z_reg
        else
            zpos[i + 1] = 12*i - 9*TT + 6
        end
    end

    # ----------------------------------------------------------------
    # SC constraint ordering (rows 6T+1..7T):
    # Fwd I (half..TT-1) sorted desc by X Julia position,
    # then back I (half-1..1) in descending order, then I=TT.
    # ----------------------------------------------------------------
    fwd_sorted = sort(collect(half:(TT - 1)), by = i -> xpos[i + 1], rev = true)
    sc_order   = vcat(fwd_sorted, collect((half - 1):-1:1), [TT])

    # ----------------------------------------------------------------
    # Variable bounds and start point
    # ----------------------------------------------------------------
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf),  nv)

    lvar[x0_idx] = T(0); uvar[x0_idx] = T(0)
    lvar[y0_idx] = T(0); uvar[y0_idx] = T(0)
    lvar[z0_idx] = T(1); uvar[z0_idx] = T(1)
    lvar[vxI(0)] = T(0); uvar[vxI(0)] = T(0)
    lvar[vyI(0)] = T(0); uvar[vyI(0)] = T(0)
    lvar[vzI(0)] = T(0); uvar[vzI(0)] = T(0)
    lvar[vxI(TT)] = T(0); uvar[vxI(TT)] = T(0)
    lvar[vyI(TT)] = T(0); uvar[vyI(TT)] = T(0)
    lvar[vzI(TT)] = T(0); uvar[vzI(TT)] = T(0)

    for i in 1:TT
        lvar[xpos[i + 1]] = T(0)
        uvar[xpos[i + 1]] = XT
    end

    for i in 1:TT
        lvar[uxI(i)] = -FMAX; uvar[uxI(i)] = FMAX
        lvar[uyI(i)] = -FMAX; uvar[uyI(i)] = FMAX
        lvar[uzI(i)] = -FMAX; uvar[uzI(i)] = FMAX
    end

    start = zeros(T, nv)
    start[x0_idx] = T(0)
    start[y0_idx] = T(0)
    start[z0_idx] = T(1)
    # SIF start loop DO I=1..T: X(I)=I*H, VX(I)=1 (includes I=TT)
    for i in 1:TT
        start[xpos[i + 1]] = T(i) * H
        start[vxI(i)]      = T(1)
    end

    # ----------------------------------------------------------------
    # Build ExaModel
    # ----------------------------------------------------------------
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: sum_{I=1}^{T} (X(I)-XT)^2 * (I*H) / W
    obj_data = [(xi = xpos[i + 1], tiow = T(i) * H / W) for i in 1:TT]
    ExaModels.@add_obj(c, (x[d.xi] - XT)^2 * d.tiow for d in obj_data)

    # ----------------------------------------------------------------
    # Equality constraints (6*TT rows), all of the form:
    #   coef * x[b1] - coef * x[b2] - x[c3] = 0
    #
    # Build a flat ordered array of (coef, b1, b2, c3) named tuples
    # matching CUTEst's constraint ordering exactly.
    # ----------------------------------------------------------------

    # Helper to build one 6-row block for index I
    function make_block(I)
        return [
            # ACX: MoH * VX(I) - MoH * VX(I-1) - UX(I)
            (coef = MoH, b1 = vxI(I),       b2 = vxI(I - 1),   c3 = uxI(I)),
            # ACY: MoH * VY(I) - MoH * VY(I-1) - UY(I)
            (coef = MoH, b1 = vyI(I),       b2 = vyI(I - 1),   c3 = uyI(I)),
            # ACZ: MoH * VZ(I) - MoH * VZ(I-1) - UZ(I)
            (coef = MoH, b1 = vzI(I),       b2 = vzI(I - 1),   c3 = uzI(I)),
            # PSX: invH * X(I) - invH * X(I-1) - VX(I)
            (coef = invH, b1 = xpos[I + 1], b2 = xpos[I],      c3 = vxI(I)),
            # PSY: invH * Y(I) - invH * Y(I-1) - VY(I)
            (coef = invH, b1 = ypos[I + 1], b2 = ypos[I],      c3 = vyI(I)),
            # PSZ: invH * Z(I) - invH * Z(I-1) - VZ(I)
            (coef = invH, b1 = zpos[I + 1], b2 = zpos[I],      c3 = vzI(I)),
        ]
    end

    # General interleaving algorithm for equality constraints:
    # After emitting the first row(s) of block I, slot I (I=1..TT-2) inserts a terminal constraint.
    # Slot I: type = (I-1)%6 + 1  ∈ {PSZ,PSY,PSX,ACZ,ACY,ACX}, Iv = TT - div(I-1, 6)
    # Block I (I=2..TT-1): some of its rows were already emitted in earlier slots, so we
    # emit only the first (6 - count_in_slots(I)) rows here, where count_in_slots(I) counts
    # how many of the 6*(TT-I)+1 .. 6*(TT-I)+6 slots fall within 1..TT-2.
    function make_terminal_con(tp, Iv)
        tp == 1 && return (coef = invH, b1 = zpos[Iv + 1], b2 = zpos[Iv],  c3 = vzI(Iv))
        tp == 2 && return (coef = invH, b1 = ypos[Iv + 1], b2 = ypos[Iv],  c3 = vyI(Iv))
        tp == 3 && return (coef = invH, b1 = xpos[Iv + 1], b2 = xpos[Iv],  c3 = vxI(Iv))
        tp == 4 && return (coef = MoH,  b1 = vzI(Iv),      b2 = vzI(Iv-1), c3 = uzI(Iv))
        tp == 5 && return (coef = MoH,  b1 = vyI(Iv),      b2 = vyI(Iv-1), c3 = uyI(Iv))
        return          (coef = MoH,  b1 = vxI(Iv),      b2 = vxI(Iv-1), c3 = uxI(Iv))
    end

    # Build the full ordered equality constraint data
    # Block 1 always emits all 6 rows first
    eq_data = make_block(1)
    for I in 1:(TT - 2)
        tp  = (I - 1) % 6 + 1
        Iv  = TT - div(I - 1, 6)
        push!(eq_data, make_terminal_con(tp, Iv))
        # How many rows of block I+1 to emit: 6 minus how many of its rows are in slots
        next_I  = I + 1
        s_start = 6 * (TT - next_I) + 1
        s_end   = min(s_start + 5, TT - 2)
        count   = max(0, s_end - s_start + 1)
        rows    = 6 - count
        if rows > 0
            append!(eq_data, make_block(next_I)[1:rows])
        end
    end

    # Add all equality constraints via a single vectorized @add_con
    ExaModels.@add_con(c, d.coef * x[d.b1] - d.coef * x[d.b2] - x[d.c3] for d in eq_data)

    # ----------------------------------------------------------------
    # SC inequality constraints (rows 6T+1..7T)
    # sc_order[k] = I value for SC row k
    # ----------------------------------------------------------------
    sc_data = [(xi = xpos[sc_order[k] + 1],
                yi = ypos[sc_order[k] + 1],
                zi = zpos[sc_order[k] + 1])
               for k in 1:TT]
    ExaModels.@add_con(
        c,
        (x[d.yi] - sin(x[d.xi]))^2 + (x[d.zi] - cos(x[d.xi]))^2 - TOLSQ
        for d in sc_data;
        lcon = T(-Inf), ucon = T(0)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
