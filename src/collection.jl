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
    root = if set == "sifcollection"
        artifact"sifcollection"
    elseif set == "maros-meszaros"
        artifact"maros-meszaros"
    elseif set == "netlib-lp"
        artifact"netlib-lp"
    else
        error("the set $set is not supported (sifcollection, maros-meszaros, netlib-lp)")
    end
    return ENV["MASTSIF"] = _sif_dir(root)
end

"""
    _sif_dir(root) -> String

The directory of SIF files under an artifact root: the root itself, or its one
subdirectory that contains them.

Discovered rather than hardcoded. These artifacts wrap their contents in a
release-named directory that CHANGES between releases, and hardcoding the name
from an older CUTEst.jl ("optrove-maros-meszaros-9adfb5707b1e" where the
current artifact says "maros-meszaros-2026.6.2") produced a path that does not
exist. Refuses on ambiguity instead of picking, so a future artifact with two
such directories is an error rather than a silent choice.
"""
function _sif_dir(root::AbstractString)
    isdir(root) || error("artifact directory $root does not exist")
    _has_sif(d) = any(endswith(f, ".SIF") for f in readdir(d))
    _has_sif(root) && return root
    cands = [joinpath(root, d) for d in readdir(root)
             if isdir(joinpath(root, d)) && _has_sif(joinpath(root, d))]
    length(cands) == 1 && return only(cands)
    isempty(cands) && error("no .SIF files in $root or any of its " *
                            "subdirectories; the artifact layout changed")
    error("several subdirectories of $root contain .SIF files " *
          "($(basename.(cands))); refusing to guess which collection is meant")
end

"""
    mastsif() -> String

The active SIF directory, downloading the default collection if `MASTSIF` is
unset.

A `MASTSIF` that is set but not a directory is an ERROR, not a reason to fall
back: falling back silently served the default 1548-problem NLP collection to
a caller who had asked for Maros-Meszaros, and the only visible symptom was a
problem count that looked plausible.
"""
function mastsif()
    if haskey(ENV, "MASTSIF")
        d = ENV["MASTSIF"]
        isdir(d) || error("ENV[\"MASTSIF\"] = \"$d\" is not a directory; " *
                          "unset it or call set_mastsif()")
        return d
    end
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
