# collection.jl — locating SIF problem files.
#
# The collections are Julia artifacts (see Artifacts.toml), the same ones
# CUTEst.jl v1.4.0 declares, so both read byte-identical problem files. As in
# CUTEst.jl, ENV["MASTSIF"] names the active directory and is honored if the
# user set it before loading.

"""
    set_mastsif(set::String = "sifcollection")

Point `ENV["MASTSIF"]` at a collection of SIF problems. Supported sets, as in
CUTEst.jl: `"sifcollection"` (the CUTEst NLP test set, 1548 problems),
`"maros-meszaros"` (convex QPs), `"netlib-lp"` (LPs). The files download on
first use.
"""
function set_mastsif(set::String = "sifcollection")
    if set == "sifcollection"
        ENV["MASTSIF"] = artifact"sifcollection"
    elseif set == "maros-meszaros"
        ENV["MASTSIF"] = joinpath(artifact"maros-meszaros", "optrove-maros-meszaros-9adfb5707b1e")
    elseif set == "netlib-lp"
        ENV["MASTSIF"] = joinpath(artifact"netlib-lp", "optrove-netlib-lp-f83996fca937")
    else
        error("the set $set is not supported (sifcollection, maros-meszaros, netlib-lp)")
    end
    return ENV["MASTSIF"]
end

"The active SIF directory, downloading the default collection if unset."
function mastsif()
    haskey(ENV, "MASTSIF") && isdir(ENV["MASTSIF"]) && return ENV["MASTSIF"]
    return set_mastsif()
end

"""
    list_sif_problems() -> Vector{String}

Names (no `.SIF` extension) of every problem in the active collection.
"""
list_sif_problems() =
    sort!([splitext(f)[1] for f in readdir(mastsif()) if endswith(f, ".SIF")])

"""
    generator_sif_problems(; mastsif_folder = mastsif(), filter = name -> true)

Lazy generator over problem names satisfying `filter`, for sweeping the
collection without materializing the list.
"""
function generator_sif_problems(; mastsif_folder::AbstractString = mastsif(),
                                filter::Function = name -> true)
    return (
        n for n in (splitext(f)[1] for f in readdir(mastsif_folder) if endswith(f, ".SIF"))
        if filter(n)
    )
end

"""
    sif_problem_path(name) -> String

Absolute path of a problem's SIF file in the active collection. `name` may be
given with or without the `.SIF` extension.
"""
function sif_problem_path(name::AbstractString)
    base = endswith(name, ".SIF") ? name : name * ".SIF"
    path = joinpath(mastsif(), base)
    isfile(path) && return path
    error("problem $(splitext(base)[1]) is not in $(mastsif()); " *
          "see list_sif_problems() or set_mastsif()")
end
