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
    ref = CUTEst.CUTEstModel(name)
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
