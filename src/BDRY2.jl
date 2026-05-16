function bdry2_model(N = 499; T = Float64, backend = nothing, kwargs...)
    N1  = N + 1
    NV  = N1 + 1
    n   = NV * NV
    IS1 = N1 ÷ 4
    IS2 = 3 * (N1 ÷ 4)

    H    = T(1) / T(N1)
    HT   = H * T(5)
    HH2  = T(0.5) * H * H
    omH  = T(1) - H

    # Build complete col_of[I+1,J+1] = CUTEst column of X(I,J)
    col_of = zeros(Int, NV, NV)

    # Region 1: X(I,N1) for I=IS2..N at cols 1..IS1
    for I in IS2:N
        col_of[I+1, N1+1] = N1 - I
    end

    # N1 row: X(N1,J) at NV*N1+J+1
    for J in 0:N1
        col_of[N1+1, J+1] = NV * N1 + J + 1
    end

    # Large bridges: X(0,J=0..IS1-1) at NV*(N1-J)
    for J in 0:IS1-1
        col_of[1, J+1] = NV * (N1 - J)
    end

    # IS block simulation
    function is_reserved(c)
        c % NV == 0 && 2*NV <= c <= (IS1+1)*NV && return (:kn1, c ÷ NV - 1)
        lo1 = NV*IS1+IS1+1; hi1 = NV*IS1+IS2+1
        lo1 <= c <= hi1 && return (:is1, c - NV*IS1 - 1)
        lo2 = NV*(IS1+1)+IS1+1; hi2 = NV*(IS1+1)+IS2-1
        lo2 <= c <= hi2 && return (:is1p1, c - NV*(IS1+1) - 1)
        return nothing
    end
    cursor = Ref(N1 - IS2 + 1)
    function adv!()
        while true
            r = is_reserved(cursor[])
            r === nothing && break
            kind, val = r
            if     kind == :kn1;   col_of[val+1,   N1+1] = cursor[]
            elseif kind == :is1;   col_of[IS1+1,  val+1] = cursor[]
            else;                  col_of[IS1+2,  val+1] = cursor[]
            end
            cursor[] += 1
        end
    end
    for IS_row in IS2:-1:(IS1+2)
        if IS_row < IS2
            adv!()
            col_of[IS_row+1, N1+1] = cursor[]; cursor[] += 1
        end
        for J in IS2:-1:IS1
            adv!()
            col_of[IS_row+1, J+1] != 0 && continue
            col_of[IS_row+1, J+1] = cursor[]; cursor[] += 1
        end
    end
    adv!(); col_of[IS1+2, N1+1] = cursor[]; cursor[] += 1
    for J in IS2:-1:IS1
        adv!()
        col_of[IS1+2, J+1] != 0 && continue
        col_of[IS1+2, J+1] = cursor[]; cursor[] += 1
    end
    for J in IS1:IS2
        col_of[IS1+1, J+1] == 0 || continue
        adv!(); col_of[IS1+1, J+1] = cursor[]; cursor[] += 1
    end
    adv!()

    # Large-I non-IS: I=IS1+2..N, non-IS non-N1 J
    for I in IS1+2:N, J in 0:N
        col_of[I+1, J+1] != 0 && continue
        col_of[I+1, J+1] = NV * I + J + 1
    end
    # I=IS1+1, J=IS2+1..N: keep natural position
    for J in IS2+1:N
        col_of[IS1+2, J+1] = NV*(IS1+1) + J + 1
    end

    # Displaced vars: ascending nat.pos → descending available slot col
    slots = Int[]
    sizehint!(slots, 62750)
    for k in IS2:-1:(IS1+2)
        for c in NV*k+IS2+1:-1:NV*k+IS1+1; push!(slots, c); end
        push!(slots, NV * k)
    end
    push!(slots, NV*(IS1+2) - IS1)      # overflow (2nd lowest)
    push!(slots, NV*(IS1+2) - IS1 - 1)  # overflow (lowest)

    displaced = Tuple{Int,Int,Int}[]
    sizehint!(displaced, 62750)
    for J in IS1:N1; push!(displaced, (J+1, 0, J)); end
    for I in 1:IS1-1, J in 0:N; push!(displaced, (I*NV+J+1, I, J)); end
    for J in 0:IS1-1; push!(displaced, (IS1*NV+J+1, IS1, J)); end
    for J in IS2+1:N;  push!(displaced, (IS1*NV+J+1, IS1, J)); end
    for J in 0:IS1-1; push!(displaced, ((IS1+1)*NV+J+1, IS1+1, J)); end
    sort!(displaced)

    for (i, (_, I, J)) in enumerate(displaced)
        col_of[I+1, J+1] = slots[i]
    end

    vidx(i, j) = col_of[i+1, j+1]

    lvar = zeros(T, n)
    uvar = fill(T(10), n)
    for i in IS1:IS2, j in IS1:IS2
        uvar[vidx(i, j)] = T(3.15)
    end
    lvar[vidx(0, 0)]   = T(0);  uvar[vidx(0, 0)]   = T(0)
    lvar[vidx(N1, 0)]  = T(0);  uvar[vidx(N1, 0)]  = T(0)
    lvar[vidx(0, N1)]  = T(10); uvar[vidx(0, N1)]  = T(10)
    lvar[vidx(N1, N1)] = T(10); uvar[vidx(N1, N1)] = T(10)

    x0 = zeros(T, n)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, n; lvar = lvar, uvar = uvar, start = x0)

    obj_data = [(k = vidx(i, j),) for i in IS1:IS2 for j in IS1:IS2]
    ExaModels.@add_obj(c, HH2 * (x[d.k] - T(1))^2 for d in obj_data)

    P_data = [(
        ij  = vidx(i, j),
        ijp = vidx(i, j+1),
        ijm = vidx(i, j-1),
        ipj = vidx(i+1, j),
        imj = vidx(i-1, j),
    ) for i in 1:N for j in 1:N]
    ExaModels.@add_con(c,
        T(4)*x[d.ij] - x[d.ijp] - x[d.ijm] - x[d.ipj] - x[d.imj]
        for d in P_data)

    B_data = vcat([
        [(a = vidx(i, 0),  b = vidx(i, 1),  ca = T(1),  rhs = T(0)),
         (a = vidx(0, i),  b = vidx(1, i),  ca = omH,   rhs = HT ),
         (a = vidx(N1, i), b = vidx(N, i),  ca = omH,   rhs = HT )]
        for i in 1:N
    ]...)
    ExaModels.@add_con(c, d.ca * x[d.a] - x[d.b] + d.rhs for d in B_data)

    return ExaModels.ExaModel(c; kwargs...)
end
