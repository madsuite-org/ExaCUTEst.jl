# Compare the four HieLoW maximum-likelihood problems -- AKIVA, HIELOW, PARKCH
# and STRATEC -- as kernel-backed models on a GPU backend against the same
# kernels on the host, and both against the host ports `logit_oracle_model` and
# `parkch_stratec_model`, which `test/native_verify.jl` validates against the
# compiled Fortran. The chain is what makes a device result meaningful: a
# kernel can agree with its own CPU run and still compute the wrong
# log-likelihood.
#
# One script for four problems because they are one port: all four SIF files
# append the same `SUBROUTINE HIELOW`, and `logit_kernel_model` serves all four
# off one kernel-side log-likelihood.
#
# Run: julia --project=. test/logit_gpu_verify.jl [PROBLEM ...]
# Results append to runs/logit_gpu_verify.tsv as each problem finishes.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `LOGIT_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu` to
# compare the two host paths alone, which is all that is available on a machine
# without a device. Under `cpu` the `gpu/host` rows compare the host kernels
# with themselves and are vacuous by construction -- they read exactly 0.0 --
# and only `host/port` and `hess_vs_hprod` carry information.
#
# `LOGIT_GPU_CUTEST=1` adds the reference `CUTEstModel` as an independent
# anchor; it needs gfortran and the shared decode directory. Expensive for
# PARKCH and STRATEC: their SIFs list `G B1..B<n>` and all `H Bi Bj` entries as
# separate external calls, so one reference Hessian costs 120 (resp. 55) full
# sweeps over the sample.
#
# THE THREE COMPARISONS, and why none of them subsumes another.
#
#  1. `gpu/host` -- the device against the SAME kernels on KernelAbstractions'
#     CPU backend, so a difference is the device and not a different algorithm.
#     Both declare the same Hessian pattern, so the Hessian is compared slot by
#     slot.
#
#  2. `host/port` -- the host kernels against the host ports. That is the one
#     that reaches the Fortran-verified reference, so it is the one that
#     matters most, and here it carries extra weight: the kernels cannot reuse
#     `hielow_loglik` (it indexes two scratch vectors at a runtime node
#     number), so `_logit_record` is an independent second transcription of the
#     same Fortran value loop and this comparison is what adjudicates it.
#
#  3. `hess_vs_hprod` -- the assembled coordinate Hessian against `hprod`,
#     within one model. This is the only reading that exercises the coordinate
#     path at all: `hprod` goes through `hvp!` and never touches a declared
#     coordinate, so it cannot see a mis-declared pattern or a fill that writes
#     the wrong slot. That is how a duplicate-coordinate defect survived a
#     first verification on a BA objective -- 92% wrong assembled, exact in
#     `hprod`.
#
# VACUITY IS REPORTED, NOT COUNTED AS A PASS. Two variables and no constraints
# make it easy for an assertion to pass on nothing: every row records the
# magnitude of what it compared, and a row whose two sides are both zero, or
# either side empty, reads `VACUOUS` and fails the problem. All four problems
# are objective-only (`OUR2-RN-2-0`, `-3-0`, `-15-0`, `-10-0`), so `cons` and
# `jac` are recorded `n/a`, which is neither a pass nor a failure.
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

const TSV = joinpath(@__DIR__, "..", "runs", "logit_gpu_verify.tsv")
const PROBS = ["AKIVA", "HIELOW", "PARKCH", "STRATEC"]

# --- tolerances, with the mechanism beside each -----------------------------

# Host kernels against the host port, Float64. Both evaluate one record's
# conditional log-probabilities in the same order and to the same bits; what
# differs is the grouping of the outer sum over records (the port accumulates
# every level of every record into one running total, the kernels sum
# per-record subtotals) and the AD sweep that produced the derivatives
# (`ForwardDiff.gradient`/`hessian` over the whole sample against one
# single-partial directional derivative per record). Measured worst over the
# four problems and both points: 3.5e-14, on HIELOW's objective at `x0`, where
# 997 records each contribute two levels of about -0.69 to a total of 1009.
const TOL_HOST = 1e-13

# Device against host kernels. The reduction order is identical by
# construction -- stage one is a disjoint write and stage two sums each slot's
# partials sequentially in record order on both backends -- so nothing here is
# a summation-order effect and the only mechanism left is the device's
# `exp`/`log1p` against the host libm's, plus FMA contraction, an ulp apiece
# per record. That leaves this bound tight on purpose: the measured worst over
# the four problems and both points is 7.6e-16 (PARKCH's gradient at the
# perturbed point) and the OBJECTIVE agrees bit for bit on all four, so a
# result near 1e-14 would be reporting something other than this mechanism.
const TOL_DEV = 1e-14

# Device against host kernels at Float32, same mechanism scaled by
# eps(Float32) = 1.2e-7. Measured worst: 1.8e-9, STRATEC's `hprod`.
const TOL_F32_DEV = 1e-6

# The assembled coordinate Hessian times v against `hvp!`, within one model.
# The two reduce different quantities -- 120 second mixed derivatives then one
# matrix-vector product, against 15 directional second derivatives -- over
# entries as large as 1e7. Measured worst: 1.7e-14, STRATEC at `x0`.
const TOL_CANCEL = 1e-12

# Float32, host kernels against the host port. Not a device claim: the two
# differ by what a single-precision sum of `nobs` terms can deliver, which for
# STRATEC is 1573 additions into a total of 2.8e3 whose ulp is 2.4e-4, i.e. a
# relative 1e-5 before any cancellation in the derivatives. Arbitrating which
# side is closer needs the BigFloat value and is not what this row claims; it
# claims the two single-precision paths are the same computation. Measured
# worst: 1.0e-5, STRATEC's `hprod`.
const TOL_F32 = 1e-3

# BigFloat, host kernels against the host port. Default precision is 256 bits,
# so the same roundoff mechanism as `TOL_HOST` scaled by eps(BigFloat).
# Measured worst: 7.6e-76, PARKCH's `hprod`.
const TOL_BIG = 1e-60

# A Float64 objective against the BigFloat one, each path separately. This is
# an ACCURACY row, not an agreement row: it is what turns the `host/port`
# residual above from "two paths differ" into "by this much, and this side is
# nearer the exact value". The bound is the roundoff of a sum of up to 1573
# terms in Float64.
const TOL_ACC = 1e-13

# The same for Float32, where the two paths are 1e-5 apart and the question
# "which one is right" has no answer without this row: eps(Float32) times the
# number of records is 2e-4.
const TOL_ACC_F32 = 1e-4

# Against CUTEst, Float64. The reference's own summation order differs and its
# `G`/`H` are hand-coded recursions rather than AD. Measured worst: 6.4e-14,
# STRATEC's Hessian at `x0` -- the same figure the host port records against
# the same reference, which is what says the kernels have not moved the
# comparison. For PARKCH and STRATEC the reference Hessian is a real one (the
# classification's fourth character is 2 and the SIF's `INDIVIDUALS` block
# supplies `H`), so this row adjudicates second derivatives and is not merely a
# consistency check.
const TOL_REF = 1e-12

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "LOGIT_GPU_BACKEND", "cuda")
if WANT == "cuda"
    @eval using CUDA
end
const DEV = WANT == "cuda" ? CUDA.CUDABackend() : nothing
const DEVNAME = WANT == "cuda" ? "cuda" : "cpu"
if WANT == "cuda"
    CUDA.allowscalar(false)
    @info "device" name = CUDA.name(CUDA.device())
end

const WITH_REF = get(ENV, "LOGIT_GPU_CUTEST", "0") == "1"
if WITH_REF
    # Both packages export `list_sif_problems` and friends, so with both loaded
    # Julia binds neither of those names; nothing here uses them.
    @eval using CUTEst
    @eval using FileWatching.Pidfile: mkpidlock
end

# CUTEst decodes and compiles every problem into ONE shared directory, so
# concurrent construction collides and reports a compile failure that reads
# like a broken problem.
const LOCK = "/tmp/cutest-decode.pid"

dev(v) = ExaModels.convert_array(v, DEV)

# The kernel builder is loaded into the module here when `src/native.jl` does
# not already include it.
isdefined(ExaCUTEst, :logit_kernel_model) || Base.include(
    ExaCUTEst, joinpath(@__DIR__, "..", "src", "native", "LOGIT_gpu.jl"))

const PORT = Dict(
    "AKIVA" => ExaCUTEst.akiva_model, "HIELOW" => ExaCUTEst.hielow_model,
    "PARKCH" => ExaCUTEst.parkch_model, "STRATEC" => ExaCUTEst.stratec_model)

# The registry still routes these four to the host-loop builders, so the device
# model is pointed at the kernel builder here. Reaching it through
# `ExaCUTEstModel` is deliberate: it puts the registry wiring and the keyword
# forwarding under test alongside the arithmetic.
for p in PROBS
    ExaCUTEst.NATIVE_MODELS[p] = (; kwargs...) -> ExaCUTEst.logit_kernel_model(p; kwargs...)
end

# --- helpers ----------------------------------------------------------------

# Relative infinity-norm difference and the magnitude it was taken against.
# Returns `NaN` when either side is empty, and a zero magnitude when both sides
# are zero, so a comparison that never had anything to compare cannot be read
# as agreement.
# The quotient is formed in the ARGUMENTS' precision and only then narrowed:
# rounding a BigFloat pair to Float64 first would make two values that differ
# at the 1e-70 level compare exactly equal, which is a vacuous pass wearing a
# tight tolerance.
function rel(a, b)
    a = Array(a); b = Array(b)
    (isempty(a) || isempty(b)) && return (NaN, 0.0)
    s = max(norm(a, Inf), norm(b, Inf))
    return (s == 0 ? 0.0 : Float64(norm(a .- b, Inf) / s), Float64(s))
end

function relsc(a, b)
    s = max(abs(a), abs(b))
    return (s == 0 ? 0.0 : Float64(abs(a - b) / s), Float64(s))
end

# A model's declared Hessian pattern and values, brought to the host. The
# arrays are allocated with `similar(x, ...)` so they live wherever the model
# does and `hess_structure!` never has to index a device array from here.
function hess_coords(nlp, x)
    r = similar(x, Int, nlp.meta.nnzh)
    c = similar(x, Int, nlp.meta.nnzh)
    hess_structure!(nlp, r, c)
    return Array(r), Array(c), Array(hess_coord(nlp, x))
end

# Those coordinates as a lower triangle, then as a symmetric matvec, so a
# Hessian can be checked against the hprod of the same model without either
# side's pattern or duplicate-coordinate convention entering.
hess_matrix(n, (r, c, v)) = sparse(r, c, v, n, n)
hess_times(A, v) = A * v .+ (A - Diagonal(A))' * v

# The same triangle as a full symmetric matrix, which is what survives a
# variable permutation: permuting a lower triangle does not keep it lower
# triangular.
symmetrize(A) = A + (A - Diagonal(A))'

function relmat(A, B)
    (nnz(A) == 0 || nnz(B) == 0) && return (NaN, 0.0)
    s = max(norm(A, Inf), norm(B, Inf))
    return (s == 0 ? 0.0 : norm(A - B, Inf) / s, s)
end

# CUTEst's variable names, in CUTEst's own order.
function refvarnames(ref)
    n = ref.meta.nvar
    st = zeros(Cint, 1)
    vname = Matrix{Cchar}(undef, 10, n)
    CUTEst.varnames(Float64, ref.libsif, st, Cint[n], vname)
    return [String(strip(join(Char(UInt8(c)) for c in vname[:, j]))) for j in 1:n]
end

struct Check
    cb::String
    typ::String
    pt::String
    cmp::String
    val::Float64
    tol::Float64
    mag::Float64
    na::Bool
end
Check(cb, typ, pt, cmp, (v, m), tol) = Check(cb, typ, pt, cmp, v, tol, m, false)
na(cb, typ, pt, why) = Check(cb, typ, pt, why, NaN, NaN, 0.0, true)

function status(c::Check)
    c.na && return "n/a"
    (!isfinite(c.val) || c.mag == 0) && return "VACUOUS"
    return c.val <= c.tol ? "ok" : "BAD"
end

# --- one problem ------------------------------------------------------------

function verify(name)
    checks = Check[]
    th = @elapsed host = ExaCUTEst.logit_kernel_model(name)
    tp = @elapsed port = PORT[name]()
    tg = @elapsed gpu = ExaCUTEstModel(name; backend = DEV)
    n = host.meta.nvar
    host.meta.ncon == 0 || error("$name: expected an unconstrained problem")
    push!(checks, na("cons", "Float64", "-", "objective-only problem"))
    push!(checks, na("jac", "Float64", "-", "objective-only problem"))

    # The BigFloat pair, built once: it arbitrates both the Float64 and the
    # Float32 rows below, and carries the BigFloat `host/port` comparison.
    bigk = ExaCUTEst.logit_kernel_model(name; T = BigFloat)
    bigp = PORT[name](; T = BigFloat)

    Random.seed!(11)
    x0 = Array(host.meta.x0)

    for (pt, xh) in (("x0", copy(x0)), ("pert", x0 .+ 0.01 .* randn(n)))
        vh = randn(n)
        xd, vd = dev(xh), dev(vh)

        fh = obj(host, xh)
        gh = grad(host, xh)
        ph = hprod(host, xh, vh)
        ch = hess_coords(host, xh)
        Ah = hess_matrix(n, ch)

        # (2) host kernels against the Fortran-verified host port. Both declare
        # the dense lower triangle of the coefficient block in the same slot
        # order, so the Hessian is compared slot by slot -- stronger than
        # comparing assembled matrices, which can hide a permuted fill.
        cp = hess_coords(port, xh)
        push!(checks, Check("obj", "Float64", pt, "host/port",
            relsc(fh, obj(port, xh)), TOL_HOST))
        push!(checks, Check("grad", "Float64", pt, "host/port",
            rel(gh, grad(port, xh)), TOL_HOST))
        push!(checks, Check("hprod", "Float64", pt, "host/port",
            rel(ph, hprod(port, xh, vh)), TOL_HOST))
        push!(checks, Check("hess_structure", "Float64", pt, "host/port",
            (Float64(cp[1] != ch[1] || cp[2] != ch[2]), 1.0), 0.0))
        push!(checks, Check("hess", "Float64", pt, "host/port",
            rel(ch[3], cp[3]), TOL_HOST))

        # (1) the device against the same kernels on the host.
        cg = hess_coords(gpu, xd)
        push!(checks, Check("obj", "Float64", pt, "gpu/host",
            relsc(fh, obj(gpu, xd)), TOL_DEV))
        push!(checks, Check("grad", "Float64", pt, "gpu/host",
            rel(gh, grad(gpu, xd)), TOL_DEV))
        push!(checks, Check("hprod", "Float64", pt, "gpu/host",
            rel(ph, hprod(gpu, xd, vd)), TOL_DEV))
        push!(checks, Check("hess_structure", "Float64", pt, "gpu/host",
            (Float64(cg[1] != ch[1] || cg[2] != ch[2]), 1.0), 0.0))
        push!(checks, Check("hess", "Float64", pt, "gpu/host", rel(ch[3], cg[3]), TOL_DEV))

        # Which side of the `host/port` residual is nearer the exact value.
        # Both BigFloat objectives are also compared with each other, so a row
        # that reads "the kernels are closer" cannot be an artefact of taking
        # the kernels' own BigFloat value as the reference.
        xb = BigFloat.(xh)
        fbk = obj(bigk, xb)
        fbp = obj(bigp, xb)
        push!(checks, Check("obj", "BigFloat", pt, "host/port",
            relsc(fbk, fbp), TOL_BIG))
        push!(checks, Check("obj_vs_big", "Float64", pt, "host",
            relsc(BigFloat(fh), fbk), TOL_ACC))
        push!(checks, Check("obj_vs_big", "Float64", pt, "port",
            relsc(BigFloat(obj(port, xh)), fbk), TOL_ACC))

        # Reproducibility of the reduction, within the model under test: the
        # same call twice, compared BIT FOR BIT. Stage one writes one disjoint
        # slot per thread and stage two sums each slot sequentially in record
        # order, so neither stage can depend on how the launch was scheduled.
        # Exactly 0.0 is the claim; an atomic scatter into 2 to 15 slots would
        # land at 1e-16 here and fail this row.
        push!(checks, Check("obj", "Float64", pt, "dev/repeat",
            relsc(obj(gpu, xd), obj(gpu, xd)), 0.0))
        push!(checks, Check("grad", "Float64", pt, "dev/repeat",
            rel(grad(gpu, xd), grad(gpu, xd)), 0.0))
        push!(checks, Check("hess", "Float64", pt, "dev/repeat",
            rel(hess_coord(gpu, xd), hess_coord(gpu, xd)), 0.0))
        push!(checks, Check("hprod", "Float64", pt, "dev/repeat",
            rel(hprod(gpu, xd, vd), hprod(gpu, xd, vd)), 0.0))

        # (3) the coordinate path against first-order information, in each
        # model under test.
        push!(checks, Check("hess_vs_hprod", "Float64", pt, "host",
            rel(hess_times(Ah, vh), ph), TOL_CANCEL))
        push!(checks, Check("hess_vs_hprod", "Float64", pt, "gpu",
            rel(hess_times(hess_matrix(n, cg), vh), Array(hprod(gpu, xd, vd))),
            TOL_CANCEL))

        if WITH_REF
            mkpidlock(LOCK) do
                ref = CUTEstModel(name)
                try
                    # CUTEst reorders variables; map by SIF name, with a Dict
                    # rather than a scan per name.
                    ivn = Dict(s => j for (j, s) in
                               enumerate(ExaCUTEst.sif_variable_names(name)))
                    pv = Int[ivn[s] for s in refvarnames(ref)]
                    length(unique(pv)) == n ||
                        error("variable name map is not a permutation")
                    xr, vr = xh[pv], vh[pv]
                    push!(checks, Check("obj", "Float64", pt, "host/cutest",
                        relsc(fh, obj(ref, xr)), TOL_REF))
                    push!(checks, Check("grad", "Float64", pt, "host/cutest",
                        rel(gh[pv], grad(ref, xr)), TOL_REF))
                    push!(checks, Check("hprod", "Float64", pt, "host/cutest",
                        rel(ph[pv], hprod(ref, xr, vr)), TOL_REF))
                    Ar = let (rr, rc) = hess_structure(ref)
                        sparse(rr, rc, hess_coord(ref, xr), n, n)
                    end
                    push!(checks, Check("hess", "Float64", pt, "host/cutest",
                        relmat(symmetrize(Ah)[pv, pv], symmetrize(Ar)), TOL_REF))
                finally
                    finalize(ref)
                end
            end
        end
    end

    # --- element types other than Float64.
    #
    # Float32 and BigFloat both run through the kernels: the two-stage
    # reduction uses no `KernelAbstractions.@atomic`, which is what would have
    # refused a non-bits element type. BigFloat has no device array, so its row
    # is host kernels against the host port only.
    for (T, tol) in ((Float32, TOL_F32), (BigFloat, TOL_BIG))
        k = T == BigFloat ? bigk : ExaCUTEst.logit_kernel_model(name; T = T)
        p = T == BigFloat ? bigp : PORT[name](; T = T)
        xt = T.(x0)
        vt = ones(T, n)
        push!(checks, Check("obj", string(T), "x0", "host/port",
            relsc(obj(k, xt), obj(p, xt)), tol))
        push!(checks, Check("grad", string(T), "x0", "host/port",
            rel(grad(k, xt), grad(p, xt)), tol))
        push!(checks, Check("hess", string(T), "x0", "host/port",
            rel(hess_coord(k, xt), hess_coord(p, xt)), tol))
        push!(checks, Check("hprod", string(T), "x0", "host/port",
            rel(hprod(k, xt, vt), hprod(p, xt, vt)), tol))
        if T == Float32
            # The 1e-5 above, resolved: each single-precision objective against
            # the BigFloat one.
            fb0 = obj(bigk, BigFloat.(x0))
            push!(checks, Check("obj_vs_big", "Float32", "x0", "host",
                relsc(BigFloat(obj(k, xt)), fb0), TOL_ACC_F32))
            push!(checks, Check("obj_vs_big", "Float32", "x0", "port",
                relsc(BigFloat(obj(p, xt)), fb0), TOL_ACC_F32))
        end
        if T == Float32 && WANT == "cuda"
            g32 = ExaCUTEst.logit_kernel_model(name; T = T, backend = DEV)
            xd32 = dev(xt)
            vd32 = dev(vt)
            push!(checks, Check("obj", string(T), "x0", "gpu/host",
                relsc(obj(k, xt), obj(g32, xd32)), TOL_F32_DEV))
            push!(checks, Check("grad", string(T), "x0", "gpu/host",
                rel(grad(k, xt), grad(g32, xd32)), TOL_F32_DEV))
            push!(checks, Check("hess", string(T), "x0", "gpu/host",
                rel(hess_coord(k, xt), hess_coord(g32, xd32)), TOL_F32_DEV))
            push!(checks, Check("hprod", string(T), "x0", "gpu/host",
                rel(hprod(k, xt, vt), hprod(g32, xd32, vd32)), TOL_F32_DEV))
        elseif T == BigFloat
            push!(checks, na("all", string(T), "x0", "no device BigFloat array"))
        end
    end
    return checks, th, tp, tg, n, host.meta.nnzh
end

# --- driver -----------------------------------------------------------------

probs = isempty(ARGS) ? PROBS : ARGS
mkpath(dirname(TSV))
isfile(TSV) || open(TSV, "w") do io
    println(io, "problem\tdevice\tnvar\tnnzh\tbuild_host_s\tbuild_port_s\t" *
        "build_dev_s\tstatus\tcallback\ttype\tpoint\tcompare\trel\ttol\tmag")
end
nfail = 0
for name in probs
    local checks, th, tp, tg, nv, nz
    try
        checks, th, tp, tg, nv, nz = verify(name)
    catch e
        msg = replace(sprint(showerror, e)[1:min(end, 200)], '\t' => ' ', '\n' => ' ')
        @printf("%-8s ERROR %s\n", name, msg)
        open(TSV, "a") do io
            println(io, "$name\t$DEVNAME\t?\t?\t?\t?\t?\tERROR\t-\t-\t-\t-\t-\t-\t$msg")
        end
        global nfail += 1
        continue
    end
    bad = filter(c -> status(c) in ("BAD", "VACUOUS"), checks)
    nna = count(c -> status(c) == "n/a", checks)
    isempty(bad) || (global nfail += 1)
    worst = maximum((c.val for c in checks if isfinite(c.val)), init = 0.0)
    @printf("%-8s %-4s n=%-3d nnzh=%-4d build %4.1f/%4.1f/%4.1fs  %d checks (%d n/a)  worst %.2e%s\n",
        name, isempty(bad) ? "PASS" : "FAIL", nv, nz, th, tp, tg,
        length(checks), nna, worst,
        isempty(bad) ? "" : "  >> " * join(
            ("$(c.cb)/$(c.typ)/$(c.cmp)/$(c.pt)=$(c.val)[$(status(c))]" for c in bad), " "))
    open(TSV, "a") do io
        for c in checks
            @printf(io, "%s\t%s\t%d\t%d\t%.1f\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%s\t%.3e\t%.1e\t%.3e\n",
                name, DEVNAME, nv, nz, th, tp, tg, status(c), c.cb, c.typ, c.pt,
                c.cmp, c.val, c.tol, c.mag)
        end
    end
    for c in checks
        @printf("  %-14s %-9s %-5s %-12s %10.3e  tol %7.1e  mag %9.2e  %s\n",
            c.cb, c.typ, c.pt, c.cmp, c.val, c.tol, c.mag, status(c))
    end
    flush(stdout)
    GC.gc()
end
@printf("\n%d/%d problems PASS on %s\n", length(probs) - nfail, length(probs), DEVNAME)
