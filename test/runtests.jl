using Test
using ExaCUTEst
using ExaModels
using NLPModels
using CUTEst
using SparseArrays
using LinearAlgebra

const BACKENDS = Any[nothing]

# ── Problem registry ─────────────────────────────────────────────────────────────
# Each entry: (CUTEst problem name, model builder taking a backend keyword)
const MODELS = [
    ("ROSENBR",  b -> rosenbr_model(; backend = b)),
    ("HS5",      b -> hs5_model(; backend = b)),
    ("TAME",     b -> tame_model(; backend = b)),
    ("ALSOTAME", b -> alsotame_model(; backend = b)),
    ("DEGDIAG",  b -> degdiag_model(; backend = b)),
]

# ── Callback test against CUTEst reference ───────────────────────────────────────
# build_model(backend) returns an ExaModel.  WrapperNLPModel brings GPU output
# back to CPU so the same comparison code works for every backend.
function test_cutest_callbacks(cutest_name, build_model)
    ref = CUTEstModel(cutest_name)
    try
        nvar = ref.meta.nvar
        ncon = ref.meta.ncon
        nnzj = ref.meta.nnzj
        nnzh = ref.meta.nnzh
        x0   = copy(ref.meta.x0)
        y0   = zeros(Float64, ncon)

        f_ref = NLPModels.obj(ref, x0)
        g_ref = NLPModels.grad(ref, x0)
        c_ref = ncon > 0 ? NLPModels.cons(ref, x0) : Float64[]

        if ncon > 0
            jr_ref, jc_ref = NLPModels.jac_structure(ref)
            jv_ref         = NLPModels.jac_coord(ref, x0)
            J_ref          = sparse(jr_ref, jc_ref, jv_ref, ncon, nvar)
        end

        hr_ref, hc_ref = NLPModels.hess_structure(ref)
        hv_ref         = NLPModels.hess_coord(ref, x0, y0)
        H_ref          = sparse(hr_ref, hc_ref, hv_ref, nvar, nvar)

        @testset "$cutest_name" begin
            @testset "backend=$(isnothing(b) ? "nothing" : b)" for b in BACKENDS
                mw = ExaModels.WrapperNLPModel(build_model(b))

                @test mw.meta.nvar == nvar
                @test mw.meta.ncon == ncon
                # nnzj/nnzh not compared: ExaModels may count duplicate (row,col)
                # entries per block; CUTEst counts unique positions.  Correctness
                # is verified below via assembled sparse matrix norms.

                @test NLPModels.obj(mw, x0)  ≈ f_ref atol = 1e-10
                @test NLPModels.grad(mw, x0) ≈ g_ref atol = 1e-10

                if ncon > 0
                    @test NLPModels.cons(mw, x0) ≈ c_ref atol = 1e-10

                    jr, jc = NLPModels.jac_structure(mw)
                    jv     = NLPModels.jac_coord(mw, x0)
                    @test norm(sparse(jr, jc, jv, ncon, nvar) - J_ref) < 1e-10
                end

                hr, hc = NLPModels.hess_structure(mw)
                hv     = NLPModels.hess_coord(mw, x0, y0)
                @test norm(sparse(hr, hc, hv, nvar, nvar) - H_ref) < 1e-10
            end
        end
    finally
        finalize(ref)
    end
end

# ── Run all ───────────────────────────────────────────────────────────────────────
@testset "ExaCUTEst" begin
    for (name, build_model) in MODELS
        test_cutest_callbacks(name, build_model)
    end
end
