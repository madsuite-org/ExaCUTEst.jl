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
    ("HS3",      b -> hs3_model(; backend = b)),
    ("HS4",      b -> hs4_model(; backend = b)),
    ("HS5",      b -> hs5_model(; backend = b)),
    ("HS9",      b -> hs9_model(; backend = b)),
    ("HS21",     b -> hs21_model(; backend = b)),
    ("TAME",     b -> tame_model(; backend = b)),
    ("ALSOTAME", b -> alsotame_model(; backend = b)),
    ("DEGDIAG",  b -> degdiag_model(; backend = b)),
    ("ZANGWIL3", b -> zangwil3_model(; backend = b)),
    ("BOOTH",    b -> booth_model(; backend = b)),
    ("HIMMELBC", b -> himmelbc_model(; backend = b)),
    ("HIMMELBD", b -> himmelbd_model(; backend = b)),
    ("RSNBRNE",  b -> rsnbrne_model(; backend = b)),
    ("BT1",      b -> bt1_model(; backend = b)),
    ("ZECEVIC2", b -> zecevic2_model(; backend = b)),
]

# ── Callback test against CUTEst reference ───────────────────────────────────────
function test_cutest_callbacks(cutest_name, build_model)
    ref = CUTEstModel(cutest_name)
    try
        nvar = ref.meta.nvar
        ncon = ref.meta.ncon
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
                # entries per block; CUTEst counts unique positions.

                @test mw.meta.x0   ≈ x0              atol = 1e-10
                @test mw.meta.lvar ≈ ref.meta.lvar   atol = 1e-10
                @test mw.meta.uvar ≈ ref.meta.uvar   atol = 1e-10

                if ncon > 0
                    @test mw.meta.lcon ≈ ref.meta.lcon atol = 1e-10
                    @test mw.meta.ucon ≈ ref.meta.ucon atol = 1e-10
                end

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
