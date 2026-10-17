# Compare the kernel-backed models of HS67, HS85 and HS88-HS92 on a GPU
# backend against the same kernels on the host, and both against the host-loop
# ports that `test/native_verify.jl` validates against the Fortran. The chain
# is what makes a device result meaningful: a kernel can agree with its own CPU
# run and still compute the wrong cascade.
#
# Run: julia --project=. test/hs_gpu_verify.jl [PROBLEM ...]
# Results append to runs/hs_gpu_verify.tsv as each problem finishes.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `HS_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu`, which
# runs the host-versus-host half alone and is all that is available on a
# machine without a device. The `cpu` run records the device half as `skipped`,
# not as a pass.
#
# THREE COMPARISONS, each covering what the others cannot.
#
#  1. `gpu/host` -- the device against the same kernels on KernelAbstractions'
#     CPU backend. Isolates the device: same algorithm, same summation order,
#     different hardware.
#  2. `host/port` -- the host kernels against the host-loop port. This is the
#     one that matters most, because the port is what was compared with the
#     compiled Fortran, so it is what connects these kernels to the reference.
#  3. `hess_vs_hprod` -- `sparse(hess_structure, hess_coord) * v` against
#     `hprod`, inside one model. The only reading that exercises the
#     COORDINATE path: `hprod` goes through the oracle's `hvp!` and cannot see
#     a mis-declared sparsity pattern or a fill that overwrites where it
#     should accumulate, which is exactly how a defect survived the BA
#     family's first verification. What it covers is the ORACLE blocks, where
#     `hess!` and `hvp!` are separate kernels: for HS88-HS92's SIMD objective
#     term ExaModels derives both readings from one Hessian buffer, so there
#     the check is not independent and `host/port` carries that term.
#
# WHY jac AND hess ARE COMPARED ASSEMBLED in comparison 2. For HS67 and HS85
# the two builders declare the same matrices in different coordinate orders --
# the kernel model a dense Jacobian by rows, the host port a packed one -- so
# the coordinate vectors are not elementwise comparable while the matrices
# they denote are. HS88-HS92's two builders happen to declare identical
# patterns, and assembling costs nothing there. Assembling also folds in the
# declared pattern, which an elementwise comparison would not test.
# Comparison 1 has the same layout on both sides, so it compares the
# coordinates directly and a wrong transfer shows up as itself.
#
# AN ASSERTION ON THESE PROBLEMS CAN PASS ON A VECTOR OF ALL ZEROS: 1 to 21
# rows over 2 to 6 variables. So every check records the magnitude of what was
# compared, and a check whose two sides are both identically zero, or has an
# empty side, is recorded as `vacuous` and is NOT counted as a pass.
include(joinpath(@__DIR__, "gpu", "activate.jl"))
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

const TSV = joinpath(@__DIR__, "..", "runs", "hs_gpu_verify.tsv")
const PROBS = ["HS67", "HS85", "HS88", "HS89", "HS90", "HS91", "HS92"]

# The kernel builder and the host-loop port for each problem. `NATIVE_MODELS`
# does not route to the kernel builders, so the device models are built by
# assigning into it below rather than by changing the package.
const KERNEL = Dict{String,Function}(
    "HS67" => (; kwargs...) -> ExaCUTEst.hs67_kernel_model(; kwargs...),
    "HS85" => (; kwargs...) -> ExaCUTEst.hs85_kernel_model(; kwargs...),
    (p => (; kwargs...) -> ExaCUTEst.hs88_92_kernel_model(p; kwargs...)
     for p in ("HS88", "HS89", "HS90", "HS91", "HS92"))...,
)
const PORT = Dict{String,Function}(
    "HS67" => (; kwargs...) -> ExaCUTEst.hs67_model(; kwargs...),
    "HS85" => (; kwargs...) -> ExaCUTEst.hs85_model(; kwargs...),
    (p => (; kwargs...) -> ExaCUTEst.hs88_92_model(p; kwargs...)
     for p in ("HS88", "HS89", "HS90", "HS91", "HS92"))...,
)

# Two builders over the same value function on the same hardware. They differ
# only in the order a row's terms are summed (dense tables against packed
# lists) and, for the objective of HS67/HS85, in whether the Hessian comes from
# one `hess!` call or from one `hvp!` per column.
const TOL_PORT = 1e-12

# The device's divisions and transcendental functions are not bit-identical to
# the host's, and these cascades are long chains over them: HS85 is ~80
# heavily divided operations, HS88-HS92 take N exponentials per rho over 30
# rhos and then a 900-term quadratic form, and HS67's two fixed-point
# iterations differentiate the ITERATES, so a last-bit difference in an
# iterate is amplified by the recursion. One ulp at the top of such a chain is
# not one ulp at the bottom; the worst measured is 3.1e-14, on HS67's
# objective gradient at x0.
#
# The bound stays three orders above that measurement rather than ten, because
# the failure it most needs to catch is loud: HS67's loops stop on an ABSOLUTE
# `> 0.001` test, so a device rounding that changes a trip count by one shifts
# the answer by about 1e-3, and a tolerance wide enough to be safe would be
# wide enough to hide that.
const TOL_DEV = 1e-11

# Second order against first order inside one model: the same AD Hessian, read
# once through the coordinate path and once through the matrix-free one.
const TOL_SELF = 1e-12

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "HS_GPU_BACKEND", "cuda")
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

# --- helpers ----------------------------------------------------------------

# Relative infinity-norm difference, with the magnitude that produced it.
# Returns `NaN` when either side is empty, so a comparison that never had
# anything to compare cannot be read as agreement.
function reldiff(a, b)
    a = Array(a)
    b = Array(b)
    (isempty(a) || isempty(b)) && return (NaN, 0.0)
    s = max(norm(a, Inf), norm(b, Inf))
    return (s == 0 ? 0.0 : norm(a .- b, Inf) / s, s)
end

reldiff(a::Real, b::Real) = reldiff([a], [b])

# Lower-triangle coordinates assembled into a symmetric matvec, so a Hessian
# can be checked against the hprod of the same model without either side's
# duplicate-coordinate convention entering.
function hess_times(n, rows, cols, vals, v)
    A = sparse(Array(rows), Array(cols), Array(vals), n, n)
    return A * v .+ (A - Diagonal(A))' * v
end

dense_coord(rows, cols, vals, m, n) =
    Array(sparse(Array(rows), Array(cols), Array(vals), m, n))

# Field order follows `reldiff`, which returns `(value, magnitude)`, so a
# check is written `Check(cb, pt, cmp, reldiff(a, b)..., tol)`.
struct Check
    cb::String
    pt::String
    cmp::String
    val::Float64
    mag::Float64        # magnitude of the larger side; 0.0 means nothing to see
    tol::Float64
end

# A check earns `ok` only if something was actually compared. `vacuous` is
# reported on its own and never counted as agreement.
vacuous(c) = !isfinite(c.val) || c.mag == 0.0
ok(c) = !vacuous(c) && c.val <= c.tol

function verify(name)
    checks = Check[]
    tb = @elapsed host = KERNEL[name](; backend = nothing)
    ref = PORT[name]()
    tg = DEV === nothing ? 0.0 : @elapsed ExaCUTEstModel(name; backend = DEV)
    gpu = DEV === nothing ? nothing : ExaCUTEstModel(name; backend = DEV)
    n, m = host.meta.nvar, host.meta.ncon
    n == ref.meta.nvar && m == ref.meta.ncon ||
        error("$name: kernel model is $n x $m, port is " *
              "$(ref.meta.nvar) x $(ref.meta.ncon)")
    Random.seed!(11)
    x0 = Array(host.meta.x0)

    # The perturbation is RELATIVE. These cascades divide by the variables
    # (HS67 by x1, HS85 by x2 and by intermediates), so an absolute step of
    # fixed size would be a different kind of point at x = 12000 than at
    # x = 0.5.
    for (pt, xh) in (("x0", copy(x0)), ("pert", x0 .* (1 .+ 1e-3 .* randn(n))))
        yh = randn(m)
        vh = randn(n)
        xd, vd, yd = dev(xh), dev(vh), dev(yh)

        fh = obj(host, xh)
        gh = grad(host, xh)
        ch = cons(host, xh)
        jh = jac_coord(host, xh)
        hh = hess_coord(host, xh, yh)
        ph = hprod(host, xh, yh, vh)
        jrh, jch = jac_structure(host)
        hrh, hch = hess_structure(host)

        # --- 2. the host kernels against the host-loop port
        push!(checks, Check("obj", pt, "host/port", reldiff(fh, obj(ref, xh))...,
                            TOL_PORT))
        push!(checks, Check("grad", pt, "host/port", reldiff(gh, grad(ref, xh))...,
                            TOL_PORT))
        push!(checks, Check("cons", pt, "host/port", reldiff(ch, cons(ref, xh))...,
                            TOL_PORT))
        push!(checks, Check("hprod", pt, "host/port",
                            reldiff(ph, hprod(ref, xh, yh, vh))..., TOL_PORT))
        jrr, jcr = jac_structure(ref)
        push!(checks, Check("jac_assembled", pt, "host/port",
                            reldiff(dense_coord(jrh, jch, jh, m, n),
                                    dense_coord(jrr, jcr, jac_coord(ref, xh), m, n))...,
                            TOL_PORT))
        hrr, hcr = hess_structure(ref)
        push!(checks, Check("hess_assembled", pt, "host/port",
                            reldiff(dense_coord(hrh, hch, hh, n, n),
                                    dense_coord(hrr, hcr, hess_coord(ref, xh, yh), n, n))...,
                            TOL_PORT))

        # --- 3. the coordinate Hessian against the matrix-free one
        push!(checks, Check("hess_vs_hprod", pt, "host",
                            reldiff(hess_times(n, hrh, hch, hh, vh), ph)...,
                            TOL_SELF))

        # --- 1. the device against the same kernels on the CPU backend
        if gpu === nothing
            continue
        end
        push!(checks, Check("obj", pt, "gpu/host", reldiff(fh, obj(gpu, xd))...,
                            TOL_DEV))
        push!(checks, Check("grad", pt, "gpu/host", reldiff(gh, grad(gpu, xd))...,
                            TOL_DEV))
        push!(checks, Check("cons", pt, "gpu/host", reldiff(ch, cons(gpu, xd))...,
                            TOL_DEV))
        push!(checks, Check("jac", pt, "gpu/host",
                            reldiff(jh, jac_coord(gpu, xd))..., TOL_DEV))
        hd = hess_coord(gpu, xd, yd)
        pd = hprod(gpu, xd, yd, vd)
        push!(checks, Check("hess", pt, "gpu/host", reldiff(hh, hd)..., TOL_DEV))
        push!(checks, Check("hprod", pt, "gpu/host", reldiff(ph, pd)..., TOL_DEV))

        # The sparsity patterns are built on the host and shipped, so a
        # disagreement here is a transfer fault rather than an arithmetic one.
        # `mag = 1.0` keeps a match from reading as vacuous: the quantity
        # compared is the patterns, which are never empty.
        rr = similar(xd, Int, gpu.meta.nnzh)
        cc = similar(xd, Int, gpu.meta.nnzh)
        hess_structure!(gpu, rr, cc)
        push!(checks, Check("hess_structure", pt, "gpu/host",
                            Float64(Array(rr) != hrh || Array(cc) != hch), 1.0, 0.0))
        jr = similar(xd, Int, gpu.meta.nnzj)
        jc = similar(xd, Int, gpu.meta.nnzj)
        jac_structure!(gpu, jr, jc)
        push!(checks, Check("jac_structure", pt, "gpu/host",
                            Float64(Array(jr) != jrh || Array(jc) != jch), 1.0, 0.0))

        # ...and comparison 3 on the device, which is where a coordinate fill
        # that only works on one thread per item would show.
        push!(checks, Check("hess_vs_hprod", pt, "gpu",
                            reldiff(hess_times(n, rr, cc, hd, vh), Array(pd))...,
                            TOL_SELF))
    end
    return checks, tb, tg, n, m
end

# --- driver -----------------------------------------------------------------

for (nm, b) in KERNEL
    ExaCUTEst.NATIVE_MODELS[nm] = b
end

probs = isempty(ARGS) ? PROBS : ARGS
mkpath(dirname(TSV))
isfile(TSV) || open(TSV, "w") do io
    println(io, "problem\tdevice\tnvar\tncon\tbuild_host_s\tbuild_dev_s\t" *
        "verdict\tcallback\tpoint\tcompare\trel\ttol\tmagnitude")
end
nfail = 0
for name in probs
    local checks, tb, tg, nv, nc
    try
        checks, tb, tg, nv, nc = verify(name)
    catch e
        @printf("%-6s ERROR %s\n", name, sprint(showerror, e)[1:min(end, 200)])
        open(TSV, "a") do io
            println(io, "$name\t$DEVNAME\t?\t?\t?\t?\tERROR\t-\t-\t-\t-\t-\t" *
                replace(sprint(showerror, e)[1:min(end, 200)], '\t' => ' ', '\n' => ' '))
        end
        global nfail += 1
        continue
    end
    bad = filter(c -> !ok(c) && !vacuous(c), checks)
    vac = filter(vacuous, checks)
    verdict = isempty(bad) ? "PASS" : "FAIL"
    isempty(bad) || (global nfail += 1)
    worst = maximum((c.val for c in checks if isfinite(c.val)); init = 0.0)
    @printf("%-6s %-4s n=%-2d m=%-3d build %4.1f/%4.1fs  %2d checks  worst %.2e%s%s\n",
        name, verdict, nv, nc, tb, tg, length(checks), worst,
        isempty(vac) ? "" : "  " * string(length(vac)) * " VACUOUS: " *
            join(("$(c.cb)/$(c.cmp)/$(c.pt)" for c in vac), " "),
        isempty(bad) ? "" : "  >> " *
            join(("$(c.cb)/$(c.cmp)/$(c.pt)=$(c.val)" for c in bad), " "))
    open(TSV, "a") do io
        for c in checks
            @printf(io, "%s\t%s\t%d\t%d\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3e\t%.1e\t%.3e\n",
                name, DEVNAME, nv, nc, tb, tg,
                vacuous(c) ? "VACUOUS" : (ok(c) ? "ok" : "BAD"),
                c.cb, c.pt, c.cmp, c.val, c.tol, c.mag)
        end
    end
    flush(stdout)
    GC.gc()
end
if DEV === nothing
    println("\nHS_GPU_BACKEND=cpu: the gpu/host half was SKIPPED, not passed.")
end
@printf("\n%d/%d problems PASS\n", length(probs) - nfail, length(probs))
