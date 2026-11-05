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
`LUKVLI*`) for `Float64` on the CPU at their default sizes, and problems that
share their code also start faster.

The setting is stored in the active project's `LocalPreferences.toml` and takes
effect in the next Julia session, whose `using ExaCUTEst` rebuilds the image.
"""
function set_precompile(flag::Bool)
    @set_preferences!("precompile" => flag)
    @info "Restart Julia for this to take effect."
    return flag
end

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

if PRECOMPILE
    @compile_workload begin
        for name in _precompile_set()
            try
                _exercise(name)
            catch err
                @warn "skipping $name in the package image" exception = err
            end
        end
    end
end
