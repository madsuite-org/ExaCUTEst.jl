# oracle.jl — the external-Fortran tier.
#
# 34 problems in the collection append hand-coded Fortran element/group
# functions to their SIF file. Those are not expressions, so ExaModelsSIF
# refuses them (ExternalFortranError) and stays free of any Fortran
# dependency. Evaluating them means calling the compiled Fortran, which is
# this package's job: CUTEst.jl compiles exactly that Fortran, and its
# callbacks are wrapped here as ExaModels oracles so one constructor covers
# the whole collection.
#
# This tier is CPU-only and gets no SIMD abstraction: an opaque Fortran
# callback has no expression graph to vectorize, so `backend` cannot apply.
# Validating it against CUTEst.jl is also circular by construction -- both
# sides call the same compiled routine -- which is why the validation census
# records these problems as ORACLE_TIER rather than PASS.

# One live CUTEstModel per problem name. The Fortran decoder keeps global
# state keyed by name, so a second construction of the SAME problem fails with
# CUTEstException(1, "memory allocation error") while the first is still open
# -- and the first is always still open here, because the oracle closures hold
# it for the model's whole life. Measured: building a problem twice failed,
# building ten different ones in a row was fine, and an earlier model kept
# evaluating correctly after a later one was built (so this is specifically a
# repeat, not cross-contamination). Reusing the handle is also semantically
# right: both models would be the same problem.
const _ORACLE_REFS = Dict{String,Any}()

function _oracle_ref(name::AbstractString)
    key = String(name)
    haskey(_ORACLE_REFS, key) && return _ORACLE_REFS[key]
    return _ORACLE_REFS[key] = CUTEst.CUTEstModel(key)
end

"""
    clear_oracle_cache()

Finalize and forget every cached external-Fortran model, releasing the Fortran
side's handles. Rarely needed: the cache holds one entry per external-Fortran
problem actually loaded, at most 34. Any `ExaModel` built from a cleared entry
stops working, so only call this when none are in use.
"""
function clear_oracle_cache()
    for (_, ref) in _ORACLE_REFS
        try
            finalize(ref)
        catch
        end
    end
    n = length(_ORACLE_REFS)
    empty!(_ORACLE_REFS)
    return n
end

"""
    cutest_oracle_model(path; T = Float64) -> ExaModel

Build an ExaModel whose objective and constraints are ExaModels oracles backed
by the SIF file's compiled Fortran. Variable and constraint order are CUTEst's
own here (not SIF declaration order), because the callbacks are CUTEst's.
"""
function cutest_oracle_model(path::AbstractString; T = Float64)
    dir = dirname(abspath(path))
    name = replace(basename(path), r"\.SIF$" => "")
    ENV["MASTSIF"] = dir
    T === Float64 || error("the external-Fortran tier is Float64-backed " *
                           "(the compiled callbacks are double precision); " *
                           "got T = $T")
    ref = _oracle_ref(name)
    meta = ref.meta
    n, m = meta.nvar, meta.ncon

    c = ExaModels.ExaCore(T)
    c, x = ExaModels.add_var(c, n; start = meta.x0, lvar = meta.lvar, uvar = meta.uvar)

    hrows, hcols = NLPModels.hess_structure(ref)
    y0 = zeros(m)
    sor = ExaModels.ScalarNonlinearOracle(;
        nvar = n,
        f = x_ -> NLPModels.obj(ref, x_),
        grad! = (g, x_) -> NLPModels.grad!(ref, x_, g),
        hvp! = (Hv, x_, v) -> NLPModels.hprod!(ref, x_, y0, v, Hv; obj_weight = 1.0),
        hess_rows = copy(hrows), hess_cols = copy(hcols),
        adapt = Val(true),
    )
    c = ExaModels.objective(c, sor)

    if m > 0
        jrows, jcols = NLPModels.jac_structure(ref)
        vor = ExaModels.VectorNonlinearOracle(;
            nvar = n, ncon = m,
            jac_rows = copy(jrows), jac_cols = copy(jcols),
            hess_rows = copy(hrows), hess_cols = copy(hcols),
            lcon = meta.lcon, ucon = meta.ucon,
            f! = (cv, x_) -> NLPModels.cons!(ref, x_, cv),
            jac! = (vals, x_) -> NLPModels.jac_coord!(ref, x_, vals),
            hess! = (vals, x_, y) -> NLPModels.hess_coord!(ref, x_, y, vals; obj_weight = 0.0),
            adapt = Val(true),
        )
        c = ExaModels.constraint(c, vor)
    end
    return ExaModels.ExaModel(c)
end
