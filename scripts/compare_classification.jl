# scripts/compare_classification.jl — validate our decode-derived
# classification table against CUTEst.jl's Fortran-derived classf.json, field
# by field over every shared problem.
#
#   julia --project=. scripts/compare_classification.jl [path/to/CUTEst/src/classf.json]
# Validate ExaCUTEst's decode-derived classification against CUTEst.jl's
# Fortran-derived classf.json, field by field over every shared problem.
using JSON
mine = JSON.parsefile(joinpath(@__DIR__, "..", "src", "classf.json"))
ref = length(ARGS) >= 1 ? ARGS[1] :
    only(filter(isfile, [joinpath(d, "src", "classf.json")
                         for d in readdir(joinpath(homedir(), ".julia", "packages", "CUTEst"),
                                          join = true)
                         if isfile(joinpath(d, "src", "classf.json")) &&
                            occursin("1fa01d4c", read(joinpath(d, "Artifacts.toml"), String))]))
theirs = JSON.parsefile(ref)
println("reference: ", ref)
shared = sort(collect(intersect(keys(mine), keys(theirs))))
println("mine=", length(mine), " theirs=", length(theirs), " shared=", length(shared))
@assert length(shared) > 1400 "nothing to compare -- check both tables loaded"

vfields = ["number", "fixed", "free", "bounded_below", "bounded_above", "bounded_both"]
cfields = ["number", "equality", "ineq_below", "ineq_above", "ineq_both", "linear", "nonlinear"]
top = ["objtype", "contype", "regular", "derivative_order", "origin", "has_internal_var"]

diffs = Dict{String,Vector{String}}()
for p in shared
    a, b = mine[p], theirs[p]
    for f in top
        a[f] == b[f] || push!(get!(() -> String[], diffs, f), p)
    end
    for f in vfields
        a["variables"][f] == b["variables"][f] ||
            push!(get!(() -> String[], diffs, "variables." * f), p)
    end
    for f in cfields
        a["constraints"][f] == b["constraints"][f] ||
            push!(get!(() -> String[], diffs, "constraints." * f), p)
    end
end
if isempty(diffs)
    println("IDENTICAL on all 19 fields across all $(length(shared)) shared problems")
else
    for (f, ps) in sort(collect(diffs); by = x -> -length(x[2]))
        println(rpad(f, 24), length(ps), " differ, e.g. ", first(ps, 4))
        for p in first(ps, 2)
            fld, key = occursin('.', f) ? split(f, '.') : ("", f)
            v1 = fld == "" ? mine[p][key] : mine[p][fld][key]
            v2 = fld == "" ? theirs[p][key] : theirs[p][fld][key]
            println("      ", p, ": ours=", v1, " CUTEst=", v2)
        end
    end
end
