using Distributed

# Spawn worker processes for parallel test execution.
# Each worker runs one problem at a time; different CUTEst shared libraries
# are loaded with RTLD_LOCAL so they have independent Fortran global state.
if nprocs() == 1
    addprocs(Sys.CPU_THREADS;
             exeflags = "--project=$(dirname(Base.active_project()))")
end

# Load packages and define shared state on every worker (including main).
@everywhere begin
    using Test
    using ExaCUTEst
    using ExaModels
    using NLPModels
    using CUTEst
    using SparseArrays
    using LinearAlgebra

    const BACKENDS = Any[nothing]

    # Problems where CUTEst's SIF file contains analytically incorrect Hessian formulas.
    # ExaModels computes the correct Hessian (verified by finite differences).
    const CUTEST_HESSIAN_BUGGY = Set{String}(["GULF", "WATSON"])

    # Problems that use external Fortran iterative procedures (SIF class OOI) with loose
    # convergence tolerances.
    const CUTEST_ITERATIVE_PROCEDURE = Set{String}(["HS67"])

    # Problems where ExaModels cannot represent per-element mixed bound types within a
    # single @add_con kernel.
    const CUTEST_MIXED_BOUNDS = Set{String}(["LUKVLI15"])

    # Problems where CUTEst's sifdecoder produces incorrect variable bounds (lvar/uvar)
    # due to a known XX-bound processing bug in the Fortran decoder.
    const CUTEST_BOUNDS_BUGGY = Set{String}([
        "FLOSP2HH", "FLOSP2HL", "FLOSP2HM", "FLOSP2TH", "FLOSP2TL", "FLOSP2TM"
    ])

    # Problems where CUTEst's Jacobian has incorrect variable indices due to fixed-variable
    # substitution artifacts in the sifdecoder.
    const CUTEST_JAC_BUGGY = Set{String}([
        "FLOSP2HH", "FLOSP2HL", "FLOSP2HM", "FLOSP2TH", "FLOSP2TL", "FLOSP2TM"
    ])

    # Problems where CUTEst's Fortran-evaluated derivatives are slightly less accurate
    # than ExaModels' AD at the reference starting point.
    const CUTEST_FLOAT_SENSITIVE = Set{String}([
        "HIMMELP1", "HIMMELP2", "HIMMELP3", "HIMMELP4", "HIMMELP5", "HIMMELP6"
    ])

    # Core test body — called on a worker with decode=false.
    function _test_cutest_body(cutest_name, build_model, cutest_args)
        ref = CUTEstModel(cutest_name, cutest_args...; decode = false)
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

            @testset "backend=$(isnothing(b) ? "nothing" : b)" for b in BACKENDS
                mw = ExaModels.WrapperNLPModel(build_model(b))

                @test mw.meta.nvar == nvar
                @test mw.meta.ncon == ncon

                @test mw.meta.x0   ≈ x0
                bounds_buggy = cutest_name in CUTEST_BOUNDS_BUGGY
                if bounds_buggy
                    @test_broken mw.meta.lvar ≈ ref.meta.lvar
                    @test_broken mw.meta.uvar ≈ ref.meta.uvar
                else
                    @test mw.meta.lvar ≈ ref.meta.lvar
                    @test mw.meta.uvar ≈ ref.meta.uvar
                end

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

                iterative  = cutest_name in CUTEST_ITERATIVE_PROCEDURE
                float_rtol = cutest_name in CUTEST_FLOAT_SENSITIVE ? 1e-5 : 1e-6
                float_atol = cutest_name in CUTEST_FLOAT_SENSITIVE ? 1e-7 : 1e-8

                if iterative
                    @test_broken NLPModels.obj(mw, x0)  ≈ f_ref  rtol=float_rtol atol=float_atol
                    @test_broken NLPModels.grad(mw, x0) ≈ g_ref  rtol=float_rtol atol=float_atol
                else
                    @test NLPModels.obj(mw, x0)  ≈ f_ref  rtol=float_rtol atol=float_atol
                    @test NLPModels.grad(mw, x0) ≈ g_ref  rtol=float_rtol atol=float_atol
                end

                if ncon > 0
                    if iterative
                        @test_broken NLPModels.cons(mw, x0) ≈ c_ref  rtol=float_rtol atol=float_atol
                    else
                        @test NLPModels.cons(mw, x0) ≈ c_ref  rtol=float_rtol atol=float_atol
                    end

                    jr, jc = NLPModels.jac_structure(mw)
                    jv     = NLPModels.jac_coord(mw, x0)
                    J_mw   = sparse(jr, jc, jv, ncon, nvar)
                    jac_buggy = cutest_name in CUTEST_JAC_BUGGY
                    if iterative || jac_buggy
                        @test_broken norm(J_mw - J_ref) < max(float_atol, float_rtol*norm(J_ref))
                    else
                        @test norm(J_mw - J_ref) < max(float_atol, float_rtol*norm(J_ref))
                    end
                end

                hr, hc = NLPModels.hess_structure(mw)
                hv     = NLPModels.hess_coord(mw, x0, y0)
                H_mw   = sparse(hr, hc, hv, nvar, nvar)
                if cutest_name in CUTEST_HESSIAN_BUGGY
                    @test_broken norm(H_mw - H_ref) < max(float_atol, float_rtol*norm(H_ref))
                elseif iterative
                    @test_broken norm(H_mw - H_ref) < max(float_atol, float_rtol*norm(H_ref))
                else
                    @test norm(H_mw - H_ref) < max(float_atol, float_rtol*norm(H_ref))
                end
            end
        finally
            finalize(ref)
        end
    end

    # Runs the full test body for one problem inside a standalone DefaultTestSet.
    # Returns the testset so the main process can record it into the top-level suite.
    function run_problem_testset(cutest_name, build_model, cutest_args)
        ts = Test.DefaultTestSet(cutest_name)
        Test.push_testset(ts)
        try
            _test_cutest_body(cutest_name, build_model, cutest_args)
        finally
            Test.pop_testset()
        end
        return ts
    end
end  # @everywhere

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
let idx = findfirst(m -> m[1] == "VARDIM", MODELS)
    if idx !== nothing
        MODELS[idx] = ("VARDIM", b -> vardim_model(10; backend = b), ("-param", "N=10"))
    end
end

# MANCINO: dense inner sum; use N=10 to keep compilation within CI time limits.
let idx = findfirst(m -> m[1] == "MANCINO", MODELS)
    if idx !== nothing
        MODELS[idx] = ("MANCINO", b -> mancino_model(10; backend = b), ("-param", "N=10"))
    end
end

# MSQRTALS: inner sum scales O(P²); use P=7.
let idx = findfirst(m -> m[1] == "MSQRTALS", MODELS)
    if idx !== nothing
        MODELS[idx] = ("MSQRTALS", b -> msqrtals_model(7; backend = b), ("-param", "P=7"))
    end
end

# POWER: single L2 group with N elements; use N=1000.
let idx = findfirst(m -> m[1] == "POWER", MODELS)
    if idx !== nothing
        MODELS[idx] = ("POWER", b -> power_model(1000; backend = b), ("-param", "N=1000"))
    end
end

# CHEBYQAD: N groups × N inner-sum terms → O(N²); use N=10.
let idx = findfirst(m -> m[1] == "CHEBYQAD", MODELS)
    if idx !== nothing
        MODELS[idx] = ("CHEBYQAD", b -> chebyqad_model(10; backend = b), ("-param", "N=10"))
    end
end

# HAGER1/2/3: default N=2500 makes CUTEst loading slow; use N=10.
for name in ("HAGER1", "HAGER2", "HAGER3")
    nm = lowercase(name)
    fn = getfield(ExaCUTEst, Symbol(nm * "_model"))
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(10; backend = b), ("-param", "N=10"))
        end
    end
end

# LUKVLE5/LUKVLI5: -param N=10 is rejected by sifdecoder; use N=100.
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

# BDRY2: default N=499 gives 251001 variables; use N=3 for testing.
let idx = findfirst(m -> m[1] == "BDRY2", MODELS)
    if idx !== nothing
        MODELS[idx] = ("BDRY2", b -> bdry2_model(3; backend = b), ("-param", "N=3"))
    end
end

# BLOCKQP1-5: default N=5000, B=10 is too large for CI; use N=10, B=5.
for (name, fn_sym) in (("BLOCKQP1", :blockqp1_model), ("BLOCKQP2", :blockqp2_model),
                        ("BLOCKQP3", :blockqp3_model), ("BLOCKQP4", :blockqp4_model),
                        ("BLOCKQP5", :blockqp5_model))
    local fn = getfield(ExaCUTEst, fn_sym)
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(10, 5; backend = b), ("-param", "N=10", "-param", "B=5"))
        end
    end
end

# VANDERM1-4: default N=100; use N=10 for testing.
for (name, fn_sym) in (("VANDERM1", :vanderm1_model), ("VANDERM2", :vanderm2_model),
                        ("VANDERM3", :vanderm3_model), ("VANDERM4", :vanderm4_model))
    local fn = getfield(ExaCUTEst, fn_sym)
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(10; backend = b), ("-param", "N=10"))
        end
    end
end

# CORKSCRW: default T=500 gives 4506 variables; use T=10 (96 variables) for testing.
let idx = findfirst(m -> m[1] == "CORKSCRW", MODELS)
    if idx !== nothing
        MODELS[idx] = ("CORKSCRW", b -> corkscrw_model(10; backend = b), ("-param", "T=10"))
    end
end

# CLPLATEC: default P=71 gives 5041 variables; use P=10 for testing.
let idx = findfirst(m -> m[1] == "CLPLATEC", MODELS)
    if idx !== nothing
        MODELS[idx] = ("CLPLATEC", b -> clplatec_model(10; backend = b), ("-param", "P=10"))
    end
end

# YATP2SQ: default N=350 gives 123200 variables; use N=10 for testing.
let idx = findfirst(m -> m[1] == "YATP2SQ", MODELS)
    if idx !== nothing
        MODELS[idx] = ("YATP2SQ", b -> yatp2sq_model(10; backend = b), ("-param", "N=10"))
    end
end

# SSBRYBND: default N=5000; use N=10 for testing.
let idx = findfirst(m -> m[1] == "SSBRYBND", MODELS)
    if idx !== nothing
        MODELS[idx] = ("SSBRYBND", b -> ssbrybnd_model(10; backend = b), ("-param", "N=10"))
    end
end

# SPMSRTLS: default M=1667; use M=34 (n=100) for testing.
let idx = findfirst(m -> m[1] == "SPMSRTLS", MODELS)
    if idx !== nothing
        MODELS[idx] = ("SPMSRTLS", b -> spmsrtls_model(34; backend = b), ("-param", "M=34"))
    end
end

# FLOSP2HH/HL/HM/TH/TL/TM: default M=15 gives 2883 variables; use M=1 for testing.
for (name, fn_name) in [("FLOSP2HH", :flosp2hh_model), ("FLOSP2HL", :flosp2hl_model),
                         ("FLOSP2HM", :flosp2hm_model), ("FLOSP2TH", :flosp2th_model),
                         ("FLOSP2TL", :flosp2tl_model), ("FLOSP2TM", :flosp2tm_model)]
    fn = getfield(ExaCUTEst, fn_name)
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(1; backend = b), ("-param", "M=1"))
        end
    end
end

# GASOIL: default NH=400; use NH=50 for testing.
let idx = findfirst(m -> m[1] == "GASOIL", MODELS)
    if idx !== nothing
        MODELS[idx] = ("GASOIL", b -> gasoil_model(50; backend = b), ("-param", "NH=50"))
    end
end

# GLIDER: default NH=400; use NH=50 for testing.
let idx = findfirst(m -> m[1] == "GLIDER", MODELS)
    if idx !== nothing
        MODELS[idx] = ("GLIDER", b -> glider_model(50; backend = b), ("-param", "NH=50"))
    end
end

# GRIDNETA-F: default L=30 gives 7564 variables; use L=2 (60 variables) for testing.
for (name, fn_sym) in (("GRIDNETA", :gridneta_model), ("GRIDNETB", :gridnetb_model),
                       ("GRIDNETC", :gridnetc_model), ("GRIDNETD", :gridnetd_model),
                       ("GRIDNETE", :gridnete_model), ("GRIDNETF", :gridnetf_model))
    local fn = getfield(ExaCUTEst, fn_sym)
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(2; backend = b), ("-param", "L=2"))
        end
    end
end

# MARINE: default NH=200; use NH=25 for testing.
let idx = findfirst(m -> m[1] == "MARINE", MODELS)
    if idx !== nothing
        MODELS[idx] = ("MARINE", b -> marine_model(25; backend = b), ("-param", "NH=25"))
    end
end

# RDW2D variants: default N=256; use N=4 for testing.
for (name, fn) in [
    ("RDW2D51F", rdw2d51f_model), ("RDW2D51U", rdw2d51u_model),
    ("RDW2D52B", rdw2d52b_model), ("RDW2D52F", rdw2d52f_model),
    ("RDW2D52U", rdw2d52u_model),
]
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(4; backend = b), ("-param", "N=4"))
        end
    end
end

# ZAMB2-8/9/10/11: auto-generated names use underscores but CUTEst uses hyphens
let idx = findfirst(m -> m[1] == "ZAMB2_8", MODELS)
    if idx !== nothing
        MODELS[idx] = ("ZAMB2-8", b -> zamb2_8_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "ZAMB2_9", MODELS)
    if idx !== nothing
        MODELS[idx] = ("ZAMB2-9", b -> zamb2_9_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "ZAMB2_10", MODELS)
    if idx !== nothing
        MODELS[idx] = ("ZAMB2-10", b -> zamb2_10_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "ZAMB2_11", MODELS)
    if idx !== nothing
        MODELS[idx] = ("ZAMB2-11", b -> zamb2_11_model(; backend = b), ())
    end
end

# Problems where function names use underscores but CUTEst SIF names use hyphens
let idx = findfirst(m -> m[1] == "CONT5_QP", MODELS)
    if idx !== nothing
        MODELS[idx] = ("CONT5-QP", b -> cont5_qp_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "CONT6_QQ", MODELS)
    if idx !== nothing
        MODELS[idx] = ("CONT6-QQ", b -> cont6_qq_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "HET_Z", MODELS)
    if idx !== nothing
        MODELS[idx] = ("HET-Z", b -> het_z_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "S277_280", MODELS)
    if idx !== nothing
        MODELS[idx] = ("S277-280", b -> s277_280_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "S316_322", MODELS)
    if idx !== nothing
        MODELS[idx] = ("S316-322", b -> s316_322_model(; backend = b), ())
    end
end
let idx = findfirst(m -> m[1] == "TRY_B", MODELS)
    if idx !== nothing
        MODELS[idx] = ("TRY-B", b -> try_b_model(; backend = b), ())
    end
end

# SPIN/SPIN2/SPIN2OP/SPINOP: default N=50 gives O(N²) constraint terms; use N=3.
for (name, fn_sym) in (("SPIN", :spin_model), ("SPIN2", :spin2_model),
                        ("SPIN2OP", :spin2op_model), ("SPINOP", :spinop_model))
    local fn = getfield(ExaCUTEst, fn_sym)
    let idx = findfirst(m -> m[1] == name, MODELS)
        if idx !== nothing
            MODELS[idx] = (name, b -> fn(3; backend = b), ("-param", "N=3"))
        end
    end
end

# ── Pre-decode all SIF problems (serial) ─────────────────────────────────────
# Decode and compile each problem's shared library once, so workers can load
# them with decode=false without write-access conflicts.
println("Pre-decoding $(length(MODELS)) SIF problems...")
for (name, _, cutest_args) in MODELS
    CUTEst.sifdecoder(name, cutest_args...)
    CUTEst.build_libsif(name)
end
println("Pre-decoding complete.")

# ── Parallel test run ─────────────────────────────────────────────────────────
# Each worker handles one problem; returns a DefaultTestSet whose results are
# merged back into the main "ExaCUTEst" testset via Test.record.
@testset "ExaCUTEst" begin
    parent_ts = Test.get_testset()
    results = pmap(MODELS; on_error = identity) do (name, build_model, cutest_args)
        run_problem_testset(name, build_model, cutest_args)
    end
    for (model, result) in zip(MODELS, results)
        name = model[1]
        if result isa Exception
            @testset "$name" begin
                @test false  # unexpected worker error — details in stderr
            end
        else
            Test.record(parent_ts, result)
        end
    end
end
