# precompile.jl — opt-in compiled problems.
#
# An ExaModel's type encodes its expressions, so every distinct problem compiles
# its own callbacks on first use. The `precompile` preference stores those
# callbacks for a fixed set of problems in this package's image. It is read at
# compile time: changing it invalidates the image, and `false`, the default,
# compiles nothing.

const PRECOMPILE = @load_preference("precompile", false)

# The Luksan-Vlcek problems. One image cannot hold the whole collection: linking
# fails near 2 GiB.
_precompile_set() = filter(startswith("LUKVL"), list_sif_problems())

"""
    set_precompile(flag::Bool)

Turn on or off compiling problems into ExaCUTEst's package image, so that they
build and evaluate in a new Julia session without compiling. The default is
`false`. With `true`, the image holds the Luksan-Vlcek problems (`LUKVLE*` and
`LUKVLI*`) at their default sizes, and problems that share their code also
start faster. The CPU code is `Float64`. A GPU package loaded in the project
(CUDA, AMDGPU, oneAPI, or Metal in `Float32`) adds that backend's code through
its extension, provided a device works when the extension precompiles.

The setting is stored in the active project's `LocalPreferences.toml` and takes
effect in the next Julia session, whose `using ExaCUTEst` rebuilds the image.
"""
function set_precompile(flag::Bool)
    @set_preferences!("precompile" => flag)
    @info "Restart Julia for this to take effect."
    return flag
end

# Calls every callback a solver uses, through `precompile_callbacks`, and the allocating
# forms of the NLPModels API. Through `invokelatest`, as from the top level, so that each is
# compiled as a standalone method.
function _exercise(name::AbstractString; T = Float64, backend = nothing)
    call = Base.invokelatest
    nlp = call(ExaCUTEstModel, name; T, backend)
    ExaModels.precompile_callbacks(nlp)
    x = copy(nlp.meta.x0)
    v = fill!(similar(x), one(eltype(x)))
    call(grad, nlp, x)
    if nlp.meta.ncon > 0
        y = fill!(similar(x, nlp.meta.ncon), one(eltype(x)))
        call(cons, nlp, x)
        call(jac_coord, nlp, x)
        call(hess_coord, nlp, x, y)
        call(hprod, nlp, x, y, v)
    else
        call(hess_coord, nlp, x)
        call(hprod, nlp, x, v)
    end
    return nlp
end

"""
    _precompile_workload(backend; T = Float64)

Build and evaluate the problems of the precompile set on `backend`. A problem
that fails is reported and skipped, so that it cannot discard the rest of the
image. The GPU backend extensions call this from their own workloads.
"""
function _precompile_workload(backend; T = Float64)
    for name in _precompile_set()
        try
            _exercise(name; T, backend)
        catch err
            @warn "skipping $name in the package image" exception = err
        end
    end
    # Entry points as callers write them; `-param` overrides give the same model types.
    name, call = first(_precompile_set()), Base.invokelatest
    if backend === nothing
        call(ExaCUTEstModel, name, "-param", "N=100")
        call(ExaCUTEstModel, name)
    else
        call(ExaCUTEstModel, name, "-param", "N=100"; backend)
        call(ExaCUTEstModel, name; backend)
    end
    return nothing
end

PRECOMPILE && @compile_workload _precompile_workload(nothing)
