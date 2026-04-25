using Test
using ExaCUTEst
using ExaModels
using NLPModels
using CUTEst
using SparseArrays
using LinearAlgebra
using Printf

const BACKENDS = Any[nothing]

# ── Problem registry (auto-generated from exported *_model symbols) ───────────
# Symbol → CUTEst name: strip "_model" suffix, uppercase
const MODELS = let
    fns = sort(filter(n -> endswith(string(n), "_model"), names(ExaCUTEst; all=false)))
    # Tuple: (cutest_name, build_model, cutest_args)
    Tuple{String, Function, Tuple}[
        (uppercase(replace(string(fn), "_model" => "")),
         let f = getfield(ExaCUTEst, fn); b -> f(; backend = b); end,
         ())
        for fn in fns
    ]
end

# VARDIM: L4 group requires 4D polynomial expansion; use N=10 ("original value" in SIF)
# The SIF default N=200 makes the 4D loop (N^4 = 1.6B terms) impractical for ExaModels.
let idx = findfirst(m -> m[1] == "VARDIM", MODELS)
    if idx !== nothing
        MODELS[idx] = ("VARDIM", b -> vardim_model(10; backend = b), ("-param", "N=10"))
    end
end

# ── Callback test against CUTEst reference ───────────────────────────────────────
function test_cutest_callbacks(cutest_name, build_model, cutest_args = ())
    ref = CUTEstModel(cutest_name, cutest_args...)
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

                @test mw.meta.x0   ≈ x0
                @test mw.meta.lvar ≈ ref.meta.lvar
                @test mw.meta.uvar ≈ ref.meta.uvar

                if ncon > 0
                    @test mw.meta.lcon ≈ ref.meta.lcon
                    @test mw.meta.ucon ≈ ref.meta.ucon
                end

                @test NLPModels.obj(mw, x0)  ≈ f_ref  rtol = 1e-6 atol = 1e-8
                @test NLPModels.grad(mw, x0) ≈ g_ref  rtol = 1e-6 atol = 1e-8

                if ncon > 0
                    @test NLPModels.cons(mw, x0) ≈ c_ref  rtol = 1e-6 atol = 1e-8

                    jr, jc = NLPModels.jac_structure(mw)
                    jv     = NLPModels.jac_coord(mw, x0)
                    J_mw   = sparse(jr, jc, jv, ncon, nvar)
                    @test norm(J_mw - J_ref) < max(1e-8, 1e-6*norm(J_ref))
                end

                hr, hc = NLPModels.hess_structure(mw)
                hv     = NLPModels.hess_coord(mw, x0, y0)
                H_mw   = sparse(hr, hc, hv, nvar, nvar)
                @test norm(H_mw - H_ref) < max(1e-8, 1e-6*norm(H_ref))
            end
        end
    finally
        finalize(ref)
    end
end

# ── Run all ───────────────────────────────────────────────────────────────────────
@testset "ExaCUTEst" begin
    n = length(MODELS)
    for (i, (name, build_model, cutest_args)) in enumerate(MODELS)
        filled = round(Int, 20 * i / n)
        bar = "[" * "=" ^ filled * " " ^ (20 - filled) * "]"
        print(@sprintf("\r%s %3d/%d  %-12s", bar, i, n, name))
        test_cutest_callbacks(name, build_model, cutest_args)
    end
    println()  # newline after final bar
end
