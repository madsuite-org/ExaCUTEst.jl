# classification.jl — problem selection, matching CUTEst.jl's keyword API.
#
# The MASTSIF classification string (see
# https://ralna.github.io/SIFDecode/html/classification/) gives objective and
# constraint type, regularity, derivative order, origin and whether internal
# variables appear. Sizes and bound/row-type counts come from decoding each
# problem, which is what scripts/build_classification.jl does to produce the
# shipped classf.json. The classification letters themselves are read from
# CLASSF.DB, shipped inside the collection artifact.

const objtypes = ["none", "constant", "linear", "quadratic", "sum_of_squares", "other"]
const classdb_objtype = Dict(
    "N" => objtypes[1], "C" => objtypes[2], "L" => objtypes[3],
    "Q" => objtypes[4], "S" => objtypes[5], "O" => objtypes[6],
)

const contypes = ["unc", "fixed_vars", "bounds", "network", "linear", "quadratic", "general"]
const classdb_contype = Dict(
    "U" => contypes[1], "X" => contypes[2], "B" => contypes[3], "N" => contypes[4],
    "L" => contypes[5], "Q" => contypes[6], "O" => contypes[7],
)

const origins = ["academic", "modelling", "real"]
const classdb_origin = Dict("A" => origins[1], "M" => origins[2], "R" => origins[3])

const _CLASSF = Ref{Union{Nothing,Dict{String,Any}}}(nothing)

"""
    classification() -> Dict{String,Any}

The classification table: one entry per problem, with the same fields
CUTEst.jl's `classf.json` carries (`objtype`, `contype`, `regular`,
`derivative_order`, `origin`, `has_internal_var`, `variables`, `constraints`).
Loaded once from the shipped `classf.json`.
"""
function classification()
    _CLASSF[] === nothing || return _CLASSF[]
    path = joinpath(@__DIR__, "classf.json")
    isfile(path) || error("classf.json is missing; regenerate it with " *
                          "scripts/build_classification.jl")
    return _CLASSF[] = JSON.parsefile(path)
end

"""
    classification(name) -> Dict{String,Any}

The classification entry for one problem.
"""
function classification(name::AbstractString)
    d = classification()
    haskey(d, name) || error("no classification entry for $name")
    return d[name]
end

"""
    parse_classification(line) -> NamedTuple

Decode a MASTSIF classification code such as `"SQR2-AN-2-2"` or the
`CLASSF.DB` form `"OUR2-AN-V-0"` into its fields. Exposed because it is the
only part of the selection machinery that needs no decoded model, so it works
on a problem file the decoder has never seen.
"""
function parse_classification(code::AbstractString)
    cl = collect(strip(code))
    length(cl) >= 7 || error("malformed classification code \"$code\"")
    return (
        objtype = classdb_objtype[string(cl[1])],
        contype = classdb_contype[string(cl[2])],
        regular = cl[3] == 'R',
        derivative_order = Int(cl[4] - '0'),
        origin = classdb_origin[string(cl[6])],
        has_internal_var = cl[7] == 'Y',
    )
end

"""
    select_sif_problems(; min_var = 1, max_var = Inf, min_con = 0, max_con = Inf,
                        objtype = *, contype = *, only_free_var = false,
                        only_bnd_var = false, only_linear_con = false,
                        only_nonlinear_con = false, only_equ_con = false,
                        only_ineq_con = false, custom_filter = x -> true)

Names of the problems matching the filters, with the same keywords and the
same semantics as `CUTEst.select_sif_problems`.

`min_con`/`max_con` count linear and nonlinear constraints and NOT variable
bounds, so `max_con = 0` still admits bound-constrained problems; use
`contype = "unc"` for problems with no constraints and no bounds at all.

`objtype` and `contype` accept a number, symbol, string, or vector of those:
objective `1..6` = none, constant, linear, quadratic, sum_of_squares, other;
constraint `1..7` = unc, fixed_vars, bounds, network, linear, quadratic,
general.

`custom_filter` receives a problem's classification entry (see
[`classification`](@ref)).

# Examples
```julia
select_sif_problems(; min_var = 10, max_var = 100, only_linear_con = true)
select_sif_problems(; max_con = 0)                      # may still have bounds
select_sif_problems(; objtype = :sum_of_squares)
```
"""
function select_sif_problems(;
    min_var = 1,
    max_var = Inf,
    min_con = 0,
    max_con = Inf,
    objtype = objtypes,
    contype = contypes,
    only_free_var::Bool = false,
    only_bnd_var::Bool = false,
    only_linear_con::Bool = false,
    only_nonlinear_con::Bool = false,
    only_equ_con::Bool = false,
    only_ineq_con::Bool = false,
    custom_filter::Function = x -> true,
)
    @assert !only_free_var || !only_bnd_var
    @assert !only_linear_con || !only_nonlinear_con
    @assert !only_equ_con || !only_ineq_con

    objtype = canonicalize_ftype(objtype, objtypes)
    contype = canonicalize_ftype(contype, contypes)
    objtype ⊆ objtypes || error("objtypes $objtype not supported")
    contype ⊆ contypes || error("contypes $contype not supported")

    data = classification()
    selection = String[]
    for p in keys(data)
        pv = data[p]["variables"]
        pc = data[p]["constraints"]
        nvar, ncon = pv["number"], pc["number"]
        if nvar < min_var ||
           nvar > max_var ||
           ncon < min_con ||
           ncon > max_con ||
           (only_free_var && pv["free"] < nvar) ||
           (only_bnd_var && pv["free"] > 0) ||
           (only_linear_con && pc["linear"] < ncon) ||
           (only_nonlinear_con && pc["linear"] > 0) ||
           (only_equ_con && pc["equality"] < ncon) ||
           (only_ineq_con && pc["equality"] > 0) ||
           !(data[p]["objtype"] in objtype) ||
           !(data[p]["contype"] in contype) ||
           !(custom_filter(data[p]))
            continue
        end
        push!(selection, p)
    end
    return sort!(selection)
end

canonicalize_ftype(reqtype::Integer, allowedtypes) = [allowedtypes[reqtype]]
canonicalize_ftype(reqtype::Symbol, allowedtypes) = [string(reqtype)]
canonicalize_ftype(reqtype::AbstractString, allowedtypes) = [reqtype]
canonicalize_ftype(reqtype::AbstractVector{T}, allowedtypes) where {T<:Integer} =
    allowedtypes[reqtype]
canonicalize_ftype(reqtype::AbstractVector{Symbol}, allowedtypes) = map(string, reqtype)
canonicalize_ftype(reqtype, allowedtypes) = reqtype

"""
    is_truly_unconstrained(name) -> Bool

Whether a problem has no constraints and no variable bounds.
"""
function is_truly_unconstrained(name::AbstractString)
    e = classification(name)
    return e["constraints"]["number"] == 0 && e["variables"]["free"] == e["variables"]["number"]
end

"""
    external_fortran_problems() -> Vector{String}

Problems whose element or group functions are hand-coded Fortran appended to
the SIF file. The decoder cannot translate them; `ExaCUTEstModel` builds each
from its native implementation (see [`native_problems`](@ref)).
"""
external_fortran_problems() = copy(_EXTERNAL_FORTRAN)
