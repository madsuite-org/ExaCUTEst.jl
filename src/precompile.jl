# precompile.jl — opt-in compiled problems.
#
# An ExaModel's type encodes its expressions, so every distinct problem compiles
# its own callbacks on first use. Listing problems in the `precompile_problems`
# preference stores those callbacks in this package's image. The list is read at
# compile time: changing it invalidates the image, and an empty list, the
# default, compiles nothing.

const PRECOMPILE_PROBLEMS = @load_preference("precompile_problems", String[])

"""
    set_precompile_problems(names)

Compile the problems `names` into ExaCUTEst's package image, so that building
and evaluating them in a new Julia session needs no compilation. An empty list
turns this off, which is the default. The preference is stored in the active
project's `LocalPreferences.toml` and takes effect in the next Julia session,
whose `using ExaCUTEst` rebuilds the image.

```julia
set_precompile_problems(["ROSENBR", "HS71"])
set_precompile_problems(filter(startswith("LUKVL"), list_sif_problems()))
set_precompile_problems(String[])
```

Image size, build time and load time grow with the number of distinct type
signatures among the problems. Problems of one signature share their code. An
image cannot exceed about 2 GiB, where linking fails, so this does not hold the
whole collection. Problems are compiled for `Float64` on the CPU at their
default sizes; a problem that fails to compile is reported and skipped.
"""
function set_precompile_problems(names)
    names = collect(String, names)
    unknown = setdiff(names, list_sif_problems())
    isempty(unknown) || throw(ArgumentError("not in the collection: $(join(unknown, ", "))"))
    @set_preferences!("precompile_problems" => names)
    @info "Restart Julia: `using ExaCUTEst` then compiles $(length(names)) problems."
    return names
end

"""
    precompile_problems() -> Vector{String}

The problems compiled into the image loaded in this session.
"""
precompile_problems() = copy(PRECOMPILE_PROBLEMS)

# Calls every callback a solver uses, on buffers of the model's own array type.
function _exercise(name::AbstractString)
    nlp = ExaCUTEstModel(name)
    x = copy(nlp.meta.x0)
    v = fill!(similar(x), one(eltype(x)))
    jac_structure!(nlp, similar(x, Int, nlp.meta.nnzj), similar(x, Int, nlp.meta.nnzj))
    hess_structure!(nlp, similar(x, Int, nlp.meta.nnzh), similar(x, Int, nlp.meta.nnzh))
    obj(nlp, x)
    grad(nlp, x)
    if nlp.meta.ncon > 0
        y = fill!(similar(x, nlp.meta.ncon), one(eltype(x)))
        cons(nlp, x)
        jac_coord(nlp, x)
        jprod(nlp, x, v)
        jtprod(nlp, x, y)
        hess_coord(nlp, x, y)
        hprod(nlp, x, y, v)
    else
        hess_coord(nlp, x)
        hprod(nlp, x, v)
    end
    return nlp
end

if !isempty(PRECOMPILE_PROBLEMS)
    @compile_workload begin
        for name in PRECOMPILE_PROBLEMS
            try
                _exercise(name)
            catch err
                @warn "skipping $name in the package image" exception = err
            end
        end
    end
end
