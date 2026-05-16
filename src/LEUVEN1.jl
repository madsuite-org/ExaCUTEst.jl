# LEUVEN1-7: KU Leuven MPC convex QP instances
# Source: OPTEC Workshop, Leuven 25-26/10/2010. SIF: Nick Gould, December 2010.
# classification QLR2-RN-*-*
#
# Structure: minimize 0.5*x'*Q*x + c'*x
#            subject to A*x = b (equality constraints, E type)
#            with variable bounds
#
# Implementation: parse the SIF file at runtime and build ExaModels model.

function _read_leuven_sif(sif_path::String)
    # Parse SIF file and return model data
    nvar      = 0
    obj_terms = Dict{Int,Float64}()    # var_idx → linear obj coeff
    cons_data = Dict{Int, Tuple{Char, Dict{Int,Float64}}}()  # con_idx → (type, var→coeff)
    con_rhs   = Dict{Int,Float64}()
    lvar_map  = Dict{Int,Float64}()
    uvar_map  = Dict{Int,Float64}()
    quad      = Tuple{Int,Int,Float64}[]   # (i,j,coeff) quadratic terms

    section = :none
    for line in eachline(sif_path)
        stripped = strip(line)
        isempty(stripped) && continue
        startswith(stripped, '*') && continue

        if stripped == "VARIABLES"
            section = :vars; continue
        elseif stripped == "GROUPS"
            section = :groups; continue
        elseif stripped == "CONSTANTS"
            section = :constants; continue
        elseif stripped == "BOUNDS"
            section = :bounds; continue
        elseif stripped == "QUADRATIC"
            section = :quadratic; continue
        elseif stripped in ("START POINT", "ENDATA", "OBJECT BOUND")
            section = :none; continue
        end

        if section == :vars
            m = match(r"^\s+X\s+(\d+)\s*$", line)
            if !isnothing(m)
                nvar = max(nvar, parse(Int, m[1]))
            end
        elseif section == :groups
            # Objective linear: N  OBJ  X  i  coeff
            m = match(r"^\s+N\s+OBJ\s+X\s+(\d+)\s+([\d.\-Ee+]+)", line)
            if !isnothing(m)
                vi = parse(Int, m[1])
                coeff = parse(Float64, m[2])
                obj_terms[vi] = get(obj_terms, vi, 0.0) + coeff
                continue
            end
            # Constraint: E/L/G  C  j  X  i  coeff
            m = match(r"^\s+([ELG])\s+C\s+(\d+)\s+X\s+(\d+)\s+([\d.\-Ee+]+)", line)
            if !isnothing(m)
                gtype = only(m[1])
                ci = parse(Int, m[2])
                vi = parse(Int, m[3])
                coeff = parse(Float64, m[4])
                if !haskey(cons_data, ci)
                    cons_data[ci] = (gtype, Dict{Int,Float64}())
                end
                cons_data[ci][2][vi] = get(cons_data[ci][2], vi, 0.0) + coeff
                continue
            end
            # Bare constraint declaration: E/L/G  C  j
            m = match(r"^\s+([ELG])\s+C\s+(\d+)\s*$", line)
            if !isnothing(m)
                gtype = only(m[1])
                ci = parse(Int, m[2])
                if !haskey(cons_data, ci)
                    cons_data[ci] = (gtype, Dict{Int,Float64}())
                end
            end
        elseif section == :constants
            m = match(r"^\s+\S+\s+C\s+(\d+)\s+([\d.\-Ee+]+)", line)
            if !isnothing(m)
                ci = parse(Int, m[1])
                con_rhs[ci] = parse(Float64, m[2])
            end
        elseif section == :bounds
            m = match(r"^\s+(LO|UP|FR|FX)\s+\S+\s+X\s+(\d+)\s*([\d.\-Ee+]+)?", line)
            if !isnothing(m)
                btype = m[1]
                vi = parse(Int, m[2])
                val = isnothing(m[3]) ? 0.0 : parse(Float64, m[3])
                if btype == "LO"
                    lvar_map[vi] = val
                elseif btype == "UP"
                    uvar_map[vi] = val
                elseif btype == "FX"
                    lvar_map[vi] = val; uvar_map[vi] = val
                end
                # FR: no change (already free from default)
            end
        elseif section == :quadratic
            m = match(r"^\s+X\s+(\d+)\s+X\s+(\d+)\s+([\d.\-Ee+]+)", line)
            if !isnothing(m)
                i = parse(Int, m[1]); j = parse(Int, m[2])
                coeff = parse(Float64, m[3])
                push!(quad, (i, j, coeff))
            end
        end
    end

    return (nvar=nvar, obj_terms=obj_terms, cons_data=cons_data,
            con_rhs=con_rhs, lvar_map=lvar_map, uvar_map=uvar_map, quad=quad)
end

function _build_leuven_model(sif_path::String; T=Float64, backend=nothing, kwargs...)
    d = _read_leuven_sif(sif_path)
    nvar = d.nvar

    # Bounds: default from SIF - LEUVEN uses FR DEFAULT then specific LO/UP
    lvar = fill(T(-Inf), nvar)
    uvar = fill(T(Inf), nvar)
    for (vi, lb) in d.lvar_map; lvar[vi] = T(lb); end
    for (vi, ub) in d.uvar_map; uvar[vi] = T(ub); end

    # Start point: all zeros (no START POINT section in LEUVEN SIF)
    start = zeros(T, nvar)

    c = ExaModels.ExaCore(T; backend=backend, concrete=Val(true))
    ExaModels.@add_var(c, x, nvar; lvar=lvar, uvar=uvar, start=start)

    # Linear objective terms: c'*x
    if !isempty(d.obj_terms)
        lin_obj = [(k=vi, v=T(coeff)) for (vi, coeff) in sort(collect(d.obj_terms))]
        ExaModels.@add_obj(c, d_.v * x[d_.k] for d_ in lin_obj)
    end

    # Quadratic objective terms: 0.5 * coeff * x_i * x_j
    # (QUADRATIC section convention: 0.5 * coeff for diagonal, coeff for off-diagonal)
    if !isempty(d.quad)
        quad_data = [(i=q[1], j=q[2], c_=T(0.5 * q[3])) for q in d.quad]
        diag_data = [(i=q.i, c_=q.c_) for q in quad_data if q.i == q.j]
        offdiag_data = [(i=q.i, j=q.j, c_=q.c_) for q in quad_data if q.i != q.j]
        if !isempty(diag_data)
            ExaModels.@add_obj(c, d_.c_ * x[d_.i]^2 for d_ in diag_data)
        end
        if !isempty(offdiag_data)
            ExaModels.@add_obj(c, d_.c_ * x[d_.i] * x[d_.j] for d_ in offdiag_data)
        end
    end

    # Constraints: sorted by constraint index for CUTEst ordering
    con_indices = sort(collect(keys(d.cons_data)))
    for ci in con_indices
        gtype, var_coeffs = d.cons_data[ci]
        rhs = T(get(d.con_rhs, ci, 0.0))
        entries = sort(collect(var_coeffs))  # (vi, coeff) sorted by vi

        if isempty(entries)
            # Empty constraint: just -rhs = 0
            if gtype == 'E'
                ExaModels.@add_con(c, -rhs)
            elseif gtype == 'L'
                ExaModels.@add_con(c, -rhs; lcon=T(-Inf), ucon=T(0))
            elseif gtype == 'G'
                ExaModels.@add_con(c, -rhs; lcon=T(0), ucon=T(Inf))
            end
        else
            # Build constraint expression as sum of linear terms - rhs
            # Use @add_par for the constraint data (indices and coefficients)
            con_d = [(k=e[1], v=T(e[2])) for e in entries]
            expr_val = sum(T(e[2]) * 0.0 for e in entries) - rhs  # dummy for type
            # We can't easily build dynamic expressions; use a helper
            if gtype == 'E'
                ExaModels.@add_con(c,
                    sum(d_.v * x[d_.k] for d_ in con_d) - rhs)
            elseif gtype == 'L'
                ExaModels.@add_con(c,
                    sum(d_.v * x[d_.k] for d_ in con_d) - rhs;
                    lcon=T(-Inf), ucon=T(0))
            elseif gtype == 'G'
                ExaModels.@add_con(c,
                    sum(d_.v * x[d_.k] for d_ in con_d) - rhs;
                    lcon=T(0), ucon=T(Inf))
            end
        end
    end

    return ExaModels.ExaModel(c; kwargs...)
end

const _LEUVEN_SIF_DIR = joinpath(@__DIR__, "..", "..", "cutest-sif")

function _leuven_sif_path(n::Int)
    # Try to find the SIF file
    candidates = [
        joinpath(dirname(@__DIR__), "cutest-sif", "LEUVEN$n.SIF"),
        joinpath(dirname(dirname(@__DIR__)), "cutest-sif", "LEUVEN$n.SIF"),
        joinpath(homedir(), "git", "cutest-sif", "LEUVEN$n.SIF"),
        get(ENV, "CUTEST_SIF_DIR", ""),
    ]
    for p in candidates
        isfile(p) && return p
    end
    # Fall back to environment variable CUTEST_SIF_PATH
    error("Cannot find LEUVEN$n.SIF. Set CUTEST_SIF_DIR environment variable.")
end

function leuven1_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(1); T=T, backend=backend, kwargs...)
end

function leuven2_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(2); T=T, backend=backend, kwargs...)
end

function leuven3_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(3); T=T, backend=backend, kwargs...)
end

function leuven4_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(4); T=T, backend=backend, kwargs...)
end

function leuven5_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(5); T=T, backend=backend, kwargs...)
end

function leuven6_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(6); T=T, backend=backend, kwargs...)
end

function leuven7_model(; T=Float64, backend=nothing, kwargs...)
    _build_leuven_model(_leuven_sif_path(7); T=T, backend=backend, kwargs...)
end
