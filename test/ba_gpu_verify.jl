# Compare the BA-L* family's kernel-backed model on a GPU backend against the
# same model on the host, and both against `ba_oracle_model` -- the host port
# that `test/ba_verify.jl` validates against the compiled Fortran. The chain is
# what makes a device result meaningful: a kernel can agree with its own CPU
# run and still compute the wrong projection.
#
# Run: julia --project=. test/ba_gpu_verify.jl [PROBLEM ...]
# Results append to runs/ba_gpu_verify.tsv as each problem finishes.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `BA_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu` to
# compare the two host paths alone, which is all that is available on a machine
# without a GPU.
#
# WHY THE GPU-VERSUS-CPU TOLERANCE IS LOOSER THAN THE HOST-VERSUS-HOST ONE. The
# objective gradient and both Hessian-vector products scatter-accumulate with
# atomics, so their summation order is whatever the device schedules, and it
# differs run to run. Over 694346 observations sharing a few thousand camera
# variables that is a relative 1e-9, not 1e-12. The coordinate callbacks
# (`cons`, `jac_coord`, `hess_coord` of the constrained variants) have no
# summation at all -- each observation owns a disjoint slice -- and they hold
# the tight tolerance.
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

const TSV = joinpath(@__DIR__, "..", "runs", "ba_gpu_verify.tsv")
const PROBS = ["BA-L1", "BA-L1LS", "BA-L49", "BA-L49LS", "BA-L21", "BA-L21LS",
    "BA-L73", "BA-L73LS", "BA-L16", "BA-L16LS", "BA-L52", "BA-L52LS"]

# Problems whose Hessian is too large to assemble; `hess_vs_hprod` is skipped
# there and `hprod` carries the second-order claim on its own.
const BIG = 200_000

const TOL_HOST = 1e-12       # two host paths over the same data, same order
const TOL_SCATTER = 1e-8     # atomic accumulation order differs

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "BA_GPU_BACKEND", "cuda")
if WANT == "cuda"
    @eval using CUDA
end
const DEV = WANT == "cuda" ? CUDA.CUDABackend() : nothing
const DEVNAME = WANT == "cuda" ? "cuda" : "cpu"
if WANT == "cuda"
    CUDA.allowscalar(false)
    @info "device" name = CUDA.name(CUDA.device())
end

dev(v) = ExaModels.convert_array(v, DEV)
sync() = WANT == "cuda" ? CUDA.synchronize() : nothing

# --- helpers ----------------------------------------------------------------

# Relative infinity-norm difference. Returns 0.0 when both sides are zero, and
# `NaN` when either side is empty, so a comparison that never had anything to
# compare cannot be read as agreement.
function rel(a, b)
    a = Array(a); b = Array(b)
    (isempty(a) || isempty(b)) && return NaN
    s = max(norm(a, Inf), norm(b, Inf))
    return s == 0 ? 0.0 : norm(a .- b, Inf) / s
end

relsc(a, b) = (s = max(abs(a), abs(b)); s == 0 ? 0.0 : abs(a - b) / s)

# Lower-triangle coordinates assembled into a symmetric matvec, so that a
# Hessian can be checked against the hprod of the same model without either
# side's sparsity pattern or duplicate-coordinate convention entering.
function hess_times(nlp, rows, cols, vals, v)
    n = nlp.meta.nvar
    A = sparse(rows, cols, vals, n, n)
    return A * v .+ (A - Diagonal(A))' * v
end

struct Check
    cb::String
    pt::String
    cmp::String
    val::Float64
    tol::Float64
end
ok(c) = isfinite(c.val) && c.val <= c.tol

function verify(name)
    checks = Check[]
    tb = @elapsed host = ExaCUTEst.ba_kernel_model(name)
    ref = ExaCUTEst.ba_oracle_model(name)
    tg = @elapsed gpu = ExaCUTEstModel(name; backend = DEV)
    n, m = host.meta.nvar, host.meta.ncon
    Random.seed!(11)
    x0 = Array(host.meta.x0)

    for (pt, xh) in (("x0", copy(x0)), ("pert", x0 .+ 0.01 .* randn(n)))
        yh = m > 0 ? randn(m) : Float64[]
        vh = randn(n)
        xd, vd, yd = dev(xh), dev(vh), dev(yh)

        # The host kernel path is the reference side of the device comparison,
        # and `ba_oracle_model` is the reference side of its own.
        fh = obj(host, xh)
        gh = grad(host, xh)
        ph = m > 0 ? hprod(host, xh, yh, vh) : hprod(host, xh, vh)
        hh = m > 0 ? hess_coord(host, xh, yh) : hess_coord(host, xh)

        push!(checks, Check("obj", pt, "host/oracle", relsc(fh, obj(ref, xh)), TOL_HOST))
        push!(checks, Check("grad", pt, "host/oracle", rel(gh, grad(ref, xh)), TOL_HOST))
        push!(checks, Check("hprod", pt, "host/oracle",
            rel(ph, m > 0 ? hprod(ref, xh, yh, vh) : hprod(ref, xh, vh)), TOL_HOST))
        push!(checks, Check("obj", pt, "gpu/host", relsc(fh, obj(gpu, xd)), TOL_SCATTER))
        push!(checks, Check("grad", pt, "gpu/host", rel(gh, grad(gpu, xd)), TOL_SCATTER))

        if m > 0
            ch = cons(host, xh)
            jh = jac_coord(host, xh)
            push!(checks, Check("cons", pt, "host/oracle", rel(ch, cons(ref, xh)), TOL_HOST))
            push!(checks, Check("jac", pt, "host/oracle", rel(jh, jac_coord(ref, xh)), TOL_HOST))
            push!(checks, Check("hess", pt, "host/oracle",
                rel(hh, hess_coord(ref, xh, yh)), TOL_HOST))
            push!(checks, Check("cons", pt, "gpu/host", rel(ch, cons(gpu, xd)), TOL_HOST))
            push!(checks, Check("jac", pt, "gpu/host", rel(jh, jac_coord(gpu, xd)), TOL_HOST))
            push!(checks, Check("hess", pt, "gpu/host",
                rel(hh, hess_coord(gpu, xd, yd)), TOL_SCATTER))
            push!(checks, Check("hprod", pt, "gpu/host",
                rel(ph, hprod(gpu, xd, yd, vd)), TOL_SCATTER))
        else
            push!(checks, Check("hess", pt, "gpu/host",
                rel(hh, hess_coord(gpu, xd)), TOL_SCATTER))
            push!(checks, Check("hprod", pt, "gpu/host",
                rel(ph, hprod(gpu, xd, vd)), TOL_SCATTER))
            # NOT compared against the oracle: BA.jl gives the *LS variants'
            # ScalarNonlinearOracle no `hess!`, so ExaModels reconstructs its
            # Hessian one column at a time -- one host pass over every
            # observation per variable, which for BA-L49LS is 23769 passes over
            # 63686 observations. The hess_vs_hprod check below is what
            # establishes those values instead.
        end

        # Second order against first order, within the model under test: does
        # the coordinate Hessian, assembled, agree with the Hessian-vector
        # product? This is the only check on the *LS variants' hess_coord that
        # does not route through the oracle.
        if n <= BIG
            rows, cols = hess_structure(host)
            push!(checks, Check("hess_vs_hprod", pt, "host",
                rel(hess_times(host, rows, cols, hh, vh), ph), TOL_SCATTER))
        end

        # Sparsity on the device: the pattern is built on the host and shipped,
        # so a wrong one here is a transfer fault, not an arithmetic one.
        rr = similar(xd, Int, gpu.meta.nnzh)
        cc = similar(xd, Int, gpu.meta.nnzh)
        hess_structure!(gpu, rr, cc)
        rh, chh = hess_structure(host)
        push!(checks, Check("hess_structure", pt, "gpu/host",
            Float64(Array(rr) != rh || Array(cc) != chh), 0.0))
    end
    return checks, tb, tg, n, m
end

# --- driver -----------------------------------------------------------------

probs = isempty(ARGS) ? PROBS : ARGS
mkpath(dirname(TSV))
isfile(TSV) || open(TSV, "w") do io
    println(io, "problem\tdevice\tnvar\tncon\tbuild_host_s\tbuild_dev_s\t" *
        "verdict\tcallback\tpoint\tcompare\trel\ttol")
end
nfail = 0
for name in probs
    local checks, tb, tg, nv, nc
    try
        checks, tb, tg, nv, nc = verify(name)
    catch e
        @printf("%-10s ERROR %s\n", name, sprint(showerror, e)[1:min(end, 160)])
        open(TSV, "a") do io
            println(io, "$name\t$DEVNAME\t?\t?\t?\t?\tERROR\t-\t-\t-\t-\t" *
                replace(sprint(showerror, e)[1:min(end, 200)], '\t' => ' ', '\n' => ' '))
        end
        global nfail += 1
        continue
    end
    bad = filter(!ok, checks)
    verdict = isempty(bad) ? "PASS" : "FAIL"
    isempty(bad) || (global nfail += 1)
    worst = maximum(c.val for c in checks if isfinite(c.val); init = 0.0)
    @printf("%-10s %-4s n=%-7d m=%-7d build %5.1f/%5.1fs  %d checks  worst %.2e%s\n",
        name, verdict, nv, nc, tb, tg, length(checks), worst,
        isempty(bad) ? "" : "  >> " *
            join(("$(c.cb)/$(c.cmp)/$(c.pt)=$(c.val)" for c in bad), " "))
    open(TSV, "a") do io
        for c in checks
            @printf(io, "%s\t%s\t%d\t%d\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3e\t%.1e\n",
                name, DEVNAME, nv, nc, tb, tg, ok(c) ? "ok" : "BAD",
                c.cb, c.pt, c.cmp, c.val, c.tol)
        end
    end
    flush(stdout)
    GC.gc()
end
@printf("\n%d/%d problems PASS\n", length(probs) - nfail, length(probs))
