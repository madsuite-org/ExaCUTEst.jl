# Compare RAYBENDS' and BLEACHNG's kernel-backed models on a GPU backend
# against the same kernels on the host, and both against `raybends_model` /
# `bleachng_model` -- the host ports that `test/native_verify.jl` validates
# against the compiled Fortran. The chain is what makes a device result
# meaningful: a kernel can agree with its own CPU run and still compute the
# wrong travel time.
#
# Run: julia --project=. test/raybends_gpu_verify.jl [PROBLEM ...]
# Results append to runs/raybends_gpu_verify.tsv as each problem finishes.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `RAYBENDS_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu`,
# which runs the host-versus-host half alone and is all that is available on a
# machine without a device. The `cpu` run records the device half as skipped,
# not as a pass.
#
# THE REFERENCE IS THE HOST PORT, NOT CUTEst. For RAYBENDS the reference could
# serve -- its SIF supplies a full `H` section and `test/native_verify.jl`
# compares the port's whole sparse Hessian against it -- but for BLEACHNG it
# cannot: CUTEst's gradient there is wrong by 39-60% in the `QIN` columns
# (ODESSA's `DF` omits a `-y1/Vg` term) and its Hessian is the Gauss-Newton
# part built from that gradient. The header of `src/native/RAYBENDS.jl` records
# both measurements. What is tested here is that the kernels reproduce the
# ports.
#
# THREE COMPARISONS, each covering what the others cannot.
#
#  1. `gpu/host` -- the device against the same kernels on KernelAbstractions'
#     CPU backend. Isolates the device: same algorithm, same summation order,
#     different hardware.
#  2. `host/port` -- the host kernels against the host-loop port. The one that
#     matters most, because the port is what was compared with the compiled
#     Fortran. Both builders declare the SAME deduplicated Hessian pattern, in
#     the same order, so the Hessian is compared slot by slot -- the stronger
#     reading, since it would catch a fill that lands an item's values in the
#     wrong entries, which summing into a matrix can hide. The patterns are
#     compared too.
#  3. `hess_vs_hprod` -- `sparse(hess_structure, hess_coord) * v` against
#     `hprod`, inside one model. The only reading that exercises the COORDINATE
#     path: `hprod` goes through the oracle's `hvp!`, separate kernels over a
#     different gather table, and cannot see a mis-declared pattern or a fill
#     that writes the wrong entry. It is not a formality here. With the
#     deduplicated pattern, `hess!` must deliver each entry ALREADY summed over
#     the items that touch it, and RAYBENDS' first segment has
#     `X1 = X2 = X3 = X(0)`, so one of its elemental pairs lands on a diagonal
#     entry from both orderings and carries a factor of two. Getting that
#     factor wrong leaves `hprod` exact and the assembled Hessian 96% off,
#     which is what this check, and only this check, reads.
#
# A FOURTH SECTION, `variants`, checks that RAYBENDS' `single` keyword still
# reaches the kernel builder and still changes what it is supposed to: it
# selects between the Float32 and Float64 readings of four bare Fortran
# constants, and the two readings of the objective differ by 6.2e-7 at the
# point used here. The Float32 reading is the one that reproduces CUTEst; the
# other is 7.3e-7 off it at x0, which is the measurement `src/native/RAYBENDS.jl`
# records.
#
# Guarding against a vacuous pass matters here: BLEACHNG has 17 variables and
# no constraints, so a comparison can be satisfied by a vector of zeros. Every
# check records the magnitude of what it compared, and a check whose two sides
# are both identically zero, or has an empty side, is recorded as `vacuous` and
# is NOT counted as agreement.
include(joinpath(@__DIR__, "gpu", "activate.jl"))
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

# The kernel builders are loaded into the module here: `src/native.jl` does not
# include their file, so the package as installed has only the host ports.
isdefined(ExaCUTEst, :raybends_kernel_model) || Base.include(
    ExaCUTEst, joinpath(@__DIR__, "..", "src", "native", "RAYBENDS_gpu.jl"))

const TSV = joinpath(@__DIR__, "..", "runs", "raybends_gpu_verify.tsv")
const PROBS = ["RAYBENDS", "BLEACHNG"]

# The kernel builder and the host-loop port for each problem.
const KERNEL = Dict{String,Function}(
    "RAYBENDS" => (; kwargs...) -> ExaCUTEst.raybends_kernel_model(; kwargs...),
    "BLEACHNG" => (; kwargs...) -> ExaCUTEst.bleachng_kernel_model(; kwargs...),
)
const PORT = Dict{String,Function}(
    "RAYBENDS" => (; kwargs...) -> ExaCUTEst.raybends_model(; kwargs...),
    "BLEACHNG" => (; kwargs...) -> ExaCUTEst.bleachng_model(; kwargs...),
)

# --- tolerances, with the mechanism beside each -----------------------------

# Two host paths over the same items, calling the same value function on the
# same constants. What differs is the order the item contributions are summed
# in: the port accumulates into the output vector item by item, the kernels sum
# each output slot's contributions in the order of a table. RAYBENDS' knots are
# shared by four segments and BLEACHNG's `kLa` by all 1602 observations, and
# the objective is a sum of 1026 or 1602 positive terms reduced by `sum`
# against a sequential `+=`.
const TOL_PORT = 1e-12

# Device against the same kernels on the host. RAYBENDS is ten `sqrt`s and
# twenty reciprocals per segment and BLEACHNG three `exp`s plus an `expm1`
# quotient per observation, and the device's transcendental functions and its
# FMA contraction are not bit-identical to the host's.
#
# The two problems sit three orders apart under this one bound, and that gap is
# RAYBENDS' conditioning rather than the device's. A segment's length is
# `sqrt(dx^2 + dz^2)` where `dz` is a difference of two nearly equal knot
# heights, so one ulp at the top of the spline evaluation is amplified by that
# cancellation and again by the second derivative of it: the worst measured is
# 4.9e-12, on `hprod` at the perturbed point. BLEACHNG has no such step and its
# worst is 1.0e-15, on `obj` at the same point. The bound is set for the former
# and leaves about a factor of two, which is narrow; widening it would start to
# hide a real device fault in RAYBENDS rather than describe its arithmetic.
const TOL_DEV = 1e-11

# Second order against first order inside one model: the same AD Hessian read
# once through the coordinate path and once through the matrix-free one. The
# coordinate side sums duplicates through `sparse` and the matrix-free side
# through the gather table, so the two differ in summation order over the items
# meeting at a shared variable.
const TOL_SELF = 1e-11

# RAYBENDS' `single` keyword must MOVE the objective; agreement between the two
# readings would mean it was ignored. Measured against CUTEst, the Float64
# reading is 7.3e-7 off at x0 where the Float32 one is bit-identical, so the
# separation to require is well below that and well above rounding.
const TOL_SINGLE_MIN = 1e-9

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "RAYBENDS_GPU_BACKEND", "cuda")
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

# A lower triangle assembled into a symmetric matvec, so a Hessian can be
# checked against the hprod of the same model without either side's
# duplicate-coordinate convention entering.
function hess_times(n, rows, cols, vals, v)
    A = sparse(Array(rows), Array(cols), Array(vals), n, n)
    return A * v .+ (A - Diagonal(A))' * v
end

# Field order follows `reldiff`, which returns `(value, magnitude)`, so a check
# is written `Check(cb, pt, cmp, reldiff(a, b)..., tol)`.
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

# As `Check`, for a quantity that must EXCEED a bound -- a keyword whose effect
# must be visible rather than small. The recorded value is `bound / separation`
# against a tolerance of 1, so that `ok` still reads `val <= tol` and one
# verdict rule serves every row; a row at 1e-3 means the separation was a
# thousand times the bound it had to clear.
atleast(cb, pt, cmp, (val, mag), lo) =
    Check(cb, pt, cmp, mag == 0.0 ? 0.0 : lo / max(val, eps()), mag, 1.0)

function verify(name)
    checks = Check[]
    tb = @elapsed host = KERNEL[name](; backend = nothing)
    tp = @elapsed port = PORT[name]()
    tg = DEV === nothing ? 0.0 : @elapsed ExaCUTEstModel(name; backend = DEV)
    gpu = DEV === nothing ? nothing : ExaCUTEstModel(name; backend = DEV)
    n = host.meta.nvar
    (n == port.meta.nvar && host.meta.ncon == 0 && port.meta.ncon == 0) ||
        error("$name: kernel model is $n x $(host.meta.ncon), port is " *
              "$(port.meta.nvar) x $(port.meta.ncon); both must be " *
              "unconstrained and of equal size")
    @printf("%-9s nvar=%d  nnzh %d/%d  build %.1f/%.1f/%.1f s\n",
        name, n, host.meta.nnzh, port.meta.nnzh, tb, tp, tg)

    hrh, hch = hess_structure(host)
    hrp, hcp = hess_structure(port)
    # `mag = 1.0` keeps a match from reading as vacuous: what is compared is
    # the patterns, which are never empty.
    push!(checks, Check("hess_structure", "-", "host/port",
                        Float64(hrh != hrp || hch != hcp), 1.0, 0.0))

    Random.seed!(11)
    x0 = Array(host.meta.x0)

    # The perturbation is RELATIVE. BLEACHNG's variables span 1e-3 to 1e1 and
    # its `exp(-kLa t)` is sensitive at the small end, so an absolute step of
    # fixed size would be a different kind of point in each column.
    for (pt, xh) in (("x0", copy(x0)), ("pert", x0 .* (1 .+ 1e-3 .* randn(n))))
        vh = randn(n)
        xd, vd = dev(xh), dev(vh)

        fh = obj(host, xh)
        gh = grad(host, xh)
        hh = hess_coord(host, xh)
        ph = hprod(host, xh, vh)

        # --- 2. the host kernels against the host-loop port
        push!(checks, Check("obj", pt, "host/port", reldiff(fh, obj(port, xh))...,
                            TOL_PORT))
        push!(checks, Check("grad", pt, "host/port", reldiff(gh, grad(port, xh))...,
                            TOL_PORT))
        push!(checks, Check("hprod", pt, "host/port",
                            reldiff(ph, hprod(port, xh, vh))..., TOL_PORT))
        push!(checks, Check("hess", pt, "host/port",
                            reldiff(hh, hess_coord(port, xh))..., TOL_PORT))

        # --- 3. the coordinate Hessian against the matrix-free one
        push!(checks, Check("hess_vs_hprod", pt, "host",
                            reldiff(hess_times(n, hrh, hch, hh, vh), ph)...,
                            TOL_SELF))

        # --- 1. the device against the same kernels on the CPU backend
        gpu === nothing && continue
        push!(checks, Check("obj", pt, "gpu/host", reldiff(fh, obj(gpu, xd))...,
                            TOL_DEV))
        push!(checks, Check("grad", pt, "gpu/host", reldiff(gh, grad(gpu, xd))...,
                            TOL_DEV))
        hd = hess_coord(gpu, xd)
        pd = hprod(gpu, xd, vd)
        push!(checks, Check("hess", pt, "gpu/host", reldiff(hh, hd)..., TOL_DEV))
        push!(checks, Check("hprod", pt, "gpu/host", reldiff(ph, pd)..., TOL_DEV))

        # The sparsity pattern is built on the host and shipped, so a
        # disagreement here is a transfer fault rather than an arithmetic one.
        rr = similar(xd, Int, gpu.meta.nnzh)
        cc = similar(xd, Int, gpu.meta.nnzh)
        hess_structure!(gpu, rr, cc)
        push!(checks, Check("hess_structure", pt, "gpu/host",
                            Float64(Array(rr) != hrh || Array(cc) != hch), 1.0, 0.0))

        # ...and comparison 3 on the device, which is where a coordinate fill
        # that only works on one thread per item would show.
        push!(checks, Check("hess_vs_hprod", pt, "gpu",
                            reldiff(hess_times(n, rr, cc, hd, vh), Array(pd))...,
                            TOL_SELF))
    end
    return checks, tb, tp, tg, n
end

# RAYBENDS' two readings of the Fortran's bare constants, on the host alone:
# what is at stake is that the keyword reaches the kernel builder and does what
# it says, not which hardware it runs on.
function variants(name)
    name == "RAYBENDS" || return Check[]
    checks = Check[]
    dk = KERNEL[name](; single = false)
    dp = PORT[name](; single = false)
    sk = KERNEL[name]()
    Random.seed!(11)
    n = dk.meta.nvar
    x = Array(dk.meta.x0) .* (1 .+ 1e-3 .* randn(n))
    push!(checks, Check("obj", "pert", "double host/port",
                        reldiff(obj(dk, x), obj(dp, x))..., TOL_PORT))
    push!(checks, Check("grad", "pert", "double host/port",
                        reldiff(grad(dk, x), grad(dp, x))..., TOL_PORT))
    push!(checks, atleast("obj", "pert", "double/single sep",
                          reldiff(obj(dk, x), obj(sk, x)), TOL_SINGLE_MIN))
    return checks
end

# --- driver -----------------------------------------------------------------

# `NATIVE_MODELS` routes these two to the host-loop builders, so the device
# models are pointed at the kernel builders here rather than by changing the
# package. Reaching them through `ExaCUTEstModel` is deliberate: it puts the
# registry wiring and the keyword forwarding under test alongside the
# arithmetic.
for (nm, b) in KERNEL
    ExaCUTEst.NATIVE_MODELS[nm] = b
end

probs = isempty(ARGS) ? PROBS : ARGS
mkpath(dirname(TSV))
isfile(TSV) || open(TSV, "w") do io
    println(io, "problem\tdevice\tnvar\tbuild_host_s\tbuild_port_s\t" *
        "build_dev_s\tverdict\tcallback\tpoint\tcompare\trel\ttol\tmagnitude")
end
nfail = 0
for name in probs
    local checks, tb, tp, tg, nv
    try
        checks, tb, tp, tg, nv = verify(name)
        append!(checks, variants(name))
    catch e
        @printf("%-9s ERROR %s\n", name, sprint(showerror, e)[1:min(end, 300)])
        open(TSV, "a") do io
            println(io, "$name\t$DEVNAME\t?\t?\t?\t?\tERROR\t-\t-\t-\t-\t-\t" *
                replace(sprint(showerror, e)[1:min(end, 300)], '\t' => ' ', '\n' => ' '))
        end
        global nfail += 1
        continue
    end
    bad = filter(c -> !ok(c) && !vacuous(c), checks)
    vac = filter(vacuous, checks)
    verdict = isempty(bad) ? "PASS" : "FAIL"
    isempty(bad) || (global nfail += 1)
    for c in checks
        @printf("  %-14s %-5s %-20s %10.3e  tol %7.1e  mag %9.2e  %s\n",
            c.cb, c.pt, c.cmp, c.val, c.tol, c.mag,
            vacuous(c) ? "VACUOUS" : (ok(c) ? "ok" : "BAD"))
    end
    @printf("%-9s %-4s %d checks, %d bad, %d vacuous\n",
        name, verdict, length(checks), length(bad), length(vac))
    open(TSV, "a") do io
        for c in checks
            @printf(io, "%s\t%s\t%d\t%.1f\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3e\t%.1e\t%.3e\n",
                name, DEVNAME, nv, tb, tp, tg,
                vacuous(c) ? "VACUOUS" : (ok(c) ? "ok" : "BAD"),
                c.cb, c.pt, c.cmp, c.val, c.tol, c.mag)
        end
    end
    flush(stdout)
    GC.gc()
end
if DEV === nothing
    println("\nRAYBENDS_GPU_BACKEND=cpu: the gpu/host half was SKIPPED, not passed.")
end
@printf("\n%d/%d problems PASS\n", length(probs) - nfail, length(probs))
