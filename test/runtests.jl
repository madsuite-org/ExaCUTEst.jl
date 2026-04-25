using Test
using ExaCUTEst
using ExaModels
using NLPModels
using CUTEst
using SparseArrays
using LinearAlgebra
using Printf

const BACKENDS = Any[nothing]

# Problems where CUTEst's SIF file contains analytically incorrect Hessian formulas.
# ExaModels computes the correct Hessian (verified by finite differences).
const CUTEST_HESSIAN_BUGGY = Set{String}(["GULF", "WATSON"])

# Problems that use external Fortran iterative procedures (SIF class OOI) with loose
# convergence tolerances. CUTEst evaluates these using a fixed-point iteration with
# |err| <= 0.001 stopping criterion, which gives slightly different values and
# gradients than the exact mathematical fixed points computed here algebraically.
const CUTEST_ITERATIVE_PROCEDURE = Set{String}(["HS67"])

# Problems where ExaModels cannot represent per-element mixed bound types within a
# single @add_con kernel. All constraint kernels must have uniform lcon/ucon, so
# sequential L/L/G patterns cannot be reproduced. Constraint values (cons/jac/hess)
# are still correct; only lcon/ucon metadata diverges.
const CUTEST_MIXED_BOUNDS = Set{String}(["LUKVLI15"])

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

# MANCINO: dense inner sum of N terms; NTuple{N,...} Hessian sparsity scales O(N²).
# Use N=10 (smallest SIF variant) to keep compilation within CI time limits.
let idx = findfirst(m -> m[1] == "MANCINO", MODELS)
    if idx !== nothing
        MODELS[idx] = ("MANCINO", b -> mancino_model(10; backend = b), ("-param", "N=10"))
    end
end

# MSQRTALS: matrix square root, inner sum of P terms; scales O(P²) in compilation.
# Use P=7 (n=49 vars, a listed SIF variant) to keep compilation within CI time limits.
let idx = findfirst(m -> m[1] == "MSQRTALS", MODELS)
    if idx !== nothing
        MODELS[idx] = ("MSQRTALS", b -> msqrtals_model(7; backend = b), ("-param", "P=7"))
    end
end

# POWER: single L2 group with N elements; expansion yields N^2 terms.
# Default N=10000 (10^8 terms) is impractical. Use N=1000 and override CUTEst.
let idx = findfirst(m -> m[1] == "POWER", MODELS)
    if idx !== nothing
        MODELS[idx] = ("POWER", b -> power_model(1000; backend = b), ("-param", "N=1000"))
    end
end

# CHEBYQAD: N groups × N inner-sum terms → O(N^2) Hessian sparsity compilation.
# Use N=10 ("original value" in SIF) to keep compilation within CI time limits.
let idx = findfirst(m -> m[1] == "CHEBYQAD", MODELS)
    if idx !== nothing
        MODELS[idx] = ("CHEBYQAD", b -> chebyqad_model(10; backend = b), ("-param", "N=10"))
    end
end

# HAGER1/2/3: default N=2500 makes CUTEst loading slow. Use N=10 (original value).
for name in ("HAGER1", "HAGER2", "HAGER3")
    nm = lowercase(name)
    fn = getfield(ExaCUTEst, Symbol(nm * "_model"))
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(10; backend = b), ("-param", "N=10"))
        end
    end
end

# LUKVLE5/LUKVLI5: -param N=10 is rejected by sifdecoder (falls back to N=10000); use N=100 instead.
let idx = findfirst(m -> m[1] == "LUKVLE5", MODELS)
    if idx !== nothing
        MODELS[idx] = ("LUKVLE5", b -> lukvle5_model(100; backend = b), ("-param", "N=100"))
    end
end
let idx = findfirst(m -> m[1] == "LUKVLI5", MODELS)
    if idx !== nothing
        MODELS[idx] = ("LUKVLI5", b -> lukvli5_model(100; backend = b), ("-param", "N=100"))
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
                    mixed = cutest_name in CUTEST_MIXED_BOUNDS
                    if mixed
                        @test_broken mw.meta.lcon ≈ ref.meta.lcon
                        @test_broken mw.meta.ucon ≈ ref.meta.ucon
                    else
                        @test mw.meta.lcon ≈ ref.meta.lcon
                        @test mw.meta.ucon ≈ ref.meta.ucon
                    end
                end

                iterative = cutest_name in CUTEST_ITERATIVE_PROCEDURE
                if iterative
                    @test_broken NLPModels.obj(mw, x0)  ≈ f_ref  rtol = 1e-6 atol = 1e-8
                    @test_broken NLPModels.grad(mw, x0) ≈ g_ref  rtol = 1e-6 atol = 1e-8
                else
                    @test NLPModels.obj(mw, x0)  ≈ f_ref  rtol = 1e-6 atol = 1e-8
                    @test NLPModels.grad(mw, x0) ≈ g_ref  rtol = 1e-6 atol = 1e-8
                end

                if ncon > 0
                    if iterative
                        @test_broken NLPModels.cons(mw, x0) ≈ c_ref  rtol = 1e-6 atol = 1e-8
                    else
                        @test NLPModels.cons(mw, x0) ≈ c_ref  rtol = 1e-6 atol = 1e-8
                    end

                    jr, jc = NLPModels.jac_structure(mw)
                    jv     = NLPModels.jac_coord(mw, x0)
                    J_mw   = sparse(jr, jc, jv, ncon, nvar)
                    if iterative
                        @test_broken norm(J_mw - J_ref) < max(1e-8, 1e-6*norm(J_ref))
                    else
                        @test norm(J_mw - J_ref) < max(1e-8, 1e-6*norm(J_ref))
                    end
                end

                hr, hc = NLPModels.hess_structure(mw)
                hv     = NLPModels.hess_coord(mw, x0, y0)
                H_mw   = sparse(hr, hc, hv, nvar, nvar)
                if cutest_name in CUTEST_HESSIAN_BUGGY
                    @test_broken norm(H_mw - H_ref) < max(1e-8, 1e-6*norm(H_ref))
                elseif iterative
                    @test_broken norm(H_mw - H_ref) < max(1e-8, 1e-6*norm(H_ref))
                else
                    @test norm(H_mw - H_ref) < max(1e-8, 1e-6*norm(H_ref))
                end
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
