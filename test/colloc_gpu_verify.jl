# Compare CAR2's and ORBIT2's kernel-backed models on a GPU backend against the
# same kernels on the host, and both against `car2_model` / `orbit2_model` --
# the host ports that `test/native_verify.jl` validates against the compiled
# Fortran. The chain is what makes a device result meaningful: a kernel can
# agree with its own CPU run and still compute the wrong collocation defect.
#
# Run: julia --project=. test/colloc_gpu_verify.jl [PROBLEM ...]
# Results append to runs/colloc_gpu_verify.tsv as each problem finishes.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `COLLOC_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu`,
# which runs the host-versus-host half alone and is all that is available on a
# machine without a device. The `cpu` run records the device half as skipped,
# not as a pass.
#
# THE REFERENCE IS THE HOST PORT, NOT CUTEst, and for these two problems that
# is not a convenience. Their SIFs supply no `H` section (classification
# `LOR1`, fourth character 1), so CUTEst's Lagrangian Hessian is the SQR
# diagonal alone -- 1998 entries on CAR2 and nothing from any COLLOC element --
# and cannot adjudicate a Hessian. Its first derivatives can, but only to about
# 1e-8: SIFdecode stores the interval endpoints at 11 significant digits and
# `TOR = TN1 - TN` cancels two numbers up to 1000x larger, so `C2 = 3/(2 TOR)`
# -- the largest Jacobian entry in either problem -- inherits up to 9e-9.
# `src/native/COLLOC.jl` records both measurements and `test/native_verify.jl`
# is where the ports meet the reference. What is tested here is that the
# kernels reproduce those ports, which is a tight comparison rather than a
# 1e-8 one.
#
# THREE COMPARISONS, each covering what the others cannot.
#
#  1. `gpu/host` -- the device against the same kernels on KernelAbstractions'
#     CPU backend. Isolates the device: same algorithm, same summation order,
#     different hardware.
#  2. `host/port` -- the host kernels against the host-loop port. The one that
#     matters most, because the port is what was compared with the compiled
#     Fortran. Both builders declare the SAME Jacobian and Hessian patterns, in
#     the same order, so these are compared slot by slot -- the stronger
#     reading, since it would catch a fill that writes an element's values into
#     the wrong slots of its own block, which summing into a matrix can hide.
#     The patterns are compared too.
#  3. `hess_vs_hprod` -- `sparse(hess_structure, hess_coord) * v` against
#     `hprod`, inside one model. The only reading that exercises the COORDINATE
#     path: `hprod` goes through the eval's `hvp!`, a separate pair of kernels,
#     and cannot see a mis-declared sparsity pattern or a fill that overwrites
#     where it should accumulate. That is exactly how a duplicate-coordinate
#     defect survived the BA family's first verification -- 92% wrong
#     assembled, exact in `hprod`.
#
# A FOURTH SECTION, `variants`, checks that the two keywords that select
# between READINGS of the Fortran still reach the kernel builder and still
# change what they are supposed to: `ref_param_rounding = true` must move the
# row values and the Jacobian (and must agree with the port, which is the
# reading the reference uses), and `element_hessian = false` must drop the
# element Hessian to CUTEst's empty one, taking `nnzh` down to the port's.
#
# Guarding against a vacuous pass: every check records the magnitude of what it
# compared, and a check whose two sides are both identically zero, or has an
# empty side, is recorded as `vacuous` and is NOT counted as agreement.
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

# The kernel builders are loaded into the module here: `src/native.jl` does not
# include their file, so the package as installed has only the host ports.
isdefined(ExaCUTEst, :car2_kernel_model) || Base.include(
    ExaCUTEst, joinpath(@__DIR__, "..", "src", "native", "COLLOC_gpu.jl"))

const TSV = joinpath(@__DIR__, "..", "runs", "colloc_gpu_verify.tsv")
const PROBS = ["CAR2", "ORBIT2"]

# The kernel builder and the host-loop port for each problem.
const KERNEL = Dict{String,Function}(
    "CAR2" => (; kwargs...) -> ExaCUTEst.car2_kernel_model(; kwargs...),
    "ORBIT2" => (; kwargs...) -> ExaCUTEst.orbit2_kernel_model(; kwargs...),
)
const PORT = Dict{String,Function}(
    "CAR2" => (; kwargs...) -> ExaCUTEst.car2_model(; kwargs...),
    "ORBIT2" => (; kwargs...) -> ExaCUTEst.orbit2_model(; kwargs...),
)

# --- tolerances, with the mechanism beside each -----------------------------

# Two host paths over the same elements, calling the same `colloc` on the same
# element parameters. The row values and the Jacobian are elementwise
# independent, so the only difference is the order ForwardDiff seeds its
# partials in -- the port goes through `GradientConfig`/`HessianConfig`, these
# kernels seed by hand -- which changes nothing about the arithmetic. The
# Hessian-vector product differs in summation order: the port accumulates per
# element into one vector, the kernels sum each variable's contributions in the
# order of a table. Over 3996 elements all touching the free time Q that is
# still a relative 1e-13.
const TOL_PORT = 1e-12

# Device against the same kernels on the host. CAR2's defect is a handful of
# multiply-adds; ORBIT2 routes through `(x1^2+x2^2+x3^2)^1.5` and two
# divisions, and the device's `pow` and its FMA contraction are not
# bit-identical to the host's.
#
# Neither problem comes close to this bound, and the bound is not what the
# measurement shows: CAR2 agrees BIT FOR BIT on every callback at both points
# except `cons` at the perturbed one (1.9e-16), and ORBIT2's worst is 9.8e-16,
# on `hess` at the perturbed point. Collocation has no cancelling reduction --
# a row reads eleven or sixteen variables and three FSYS evaluations, and
# nothing is summed over elements except in `hprod`, where the gather table
# fixes the order on both backends. The bound is kept three orders above the
# measurement because a device fault in a kernel like this one would not be
# subtle: a wrong launch partition or a wrong slot would move an entry by
# O(1), not by an ulp.
const TOL_DEV = 1e-11

# Second order against first order inside one model: the same AD Hessian read
# once through the coordinate path and once through the matrix-free one. Both
# sum 3996 (CAR2) or 1794 (ORBIT2) element blocks, in different orders -- the
# coordinate side through `sparse`, which sums duplicates by column, and the
# matrix-free side through the gather table -- over a Lagrangian whose
# multipliers are O(1) random and whose largest entries are the C2 of the
# shortest interval.
const TOL_SELF = 1e-11

# `ref_param_rounding = true` against the port with the same keyword: again two
# host paths over identical data, so the bound is `TOL_PORT`'s. The separation
# the keyword is supposed to produce is asserted as a LOWER bound instead, at
# 1e-10: the measured effect of the rounding is ~1e-9 relative on the largest
# Jacobian entry, so a keyword silently ignored by one side would read as
# agreement at 0.0 and must not.
const TOL_ROUND_MIN = 1e-10

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "COLLOC_GPU_BACKEND", "cuda")
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
# checked against the hprod of the same model without either side's duplicate
# coordinate convention entering. `sparse` sums duplicates, which is the
# convention the blockwise pattern relies on.
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
    n, m = host.meta.nvar, host.meta.ncon
    (n == port.meta.nvar && m == port.meta.ncon) ||
        error("$name: kernel model is $n x $m, port is " *
              "$(port.meta.nvar) x $(port.meta.ncon)")
    @printf("%-7s nvar=%d ncon=%d  nnzj %d/%d  nnzh %d/%d  build %.1f/%.1f/%.1f s\n",
        name, n, m, host.meta.nnzj, port.meta.nnzj, host.meta.nnzh,
        port.meta.nnzh, tb, tp, tg)

    jrh, jch = jac_structure(host)
    hrh, hch = hess_structure(host)
    jrp, jcp = jac_structure(port)
    hrp, hcp = hess_structure(port)
    # `mag = 1.0` keeps a match from reading as vacuous: what is compared is
    # the patterns, which are never empty.
    push!(checks, Check("jac_structure", "-", "host/port",
                        Float64(jrh != jrp || jch != jcp), 1.0, 0.0))
    push!(checks, Check("hess_structure", "-", "host/port",
                        Float64(hrh != hrp || hch != hcp), 1.0, 0.0))

    Random.seed!(11)
    x0 = Array(host.meta.x0)

    for (pt, xh) in (("x0", copy(x0)), ("pert", x0 .+ 0.01 .* randn(n)))
        yh = randn(m)
        vh = randn(n)
        xd, vd, yd = dev(xh), dev(vh), dev(yh)

        fh = obj(host, xh)
        gh = grad(host, xh)
        ch = cons(host, xh)
        jh = jac_coord(host, xh)
        hh = hess_coord(host, xh, yh)
        ph = hprod(host, xh, yh, vh)

        # --- 2. the host kernels against the host-loop port
        push!(checks, Check("obj", pt, "host/port", reldiff(fh, obj(port, xh))...,
                            TOL_PORT))
        push!(checks, Check("grad", pt, "host/port", reldiff(gh, grad(port, xh))...,
                            TOL_PORT))
        push!(checks, Check("cons", pt, "host/port", reldiff(ch, cons(port, xh))...,
                            TOL_PORT))
        push!(checks, Check("jac", pt, "host/port",
                            reldiff(jh, jac_coord(port, xh))..., TOL_PORT))
        push!(checks, Check("hess", pt, "host/port",
                            reldiff(hh, hess_coord(port, xh, yh))..., TOL_PORT))
        push!(checks, Check("hprod", pt, "host/port",
                            reldiff(ph, hprod(port, xh, yh, vh))..., TOL_PORT))

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
    return checks, tb, tp, tg, n, m
end

# The two keyword readings of the Fortran, on the host alone: what is at stake
# is that the keyword reaches the kernel builder and does what it says, not
# which hardware it runs on.
function variants(name)
    checks = Check[]
    n = 0
    Random.seed!(11)

    rk = KERNEL[name](; ref_param_rounding = true)
    rp = PORT[name](; ref_param_rounding = true)
    pk = KERNEL[name]()
    n = rk.meta.nvar
    x = Array(rk.meta.x0) .+ 0.01 .* randn(n)
    push!(checks, Check("cons", "pert", "round host/port",
                        reldiff(cons(rk, x), cons(rp, x))..., TOL_PORT))
    push!(checks, Check("jac", "pert", "round host/port",
                        reldiff(jac_coord(rk, x), jac_coord(rp, x))..., TOL_PORT))
    # The keyword must MOVE the Jacobian; agreement with the unrounded build
    # would mean it was ignored.
    push!(checks, atleast("jac", "pert", "round/exact sep",
                          reldiff(jac_coord(rk, x), jac_coord(pk, x)),
                          TOL_ROUND_MIN))

    ek = KERNEL[name](; element_hessian = false)
    ep = PORT[name](; element_hessian = false)
    push!(checks, Check("nnzh", "-", "noelemhess host/port",
                        Float64(ek.meta.nnzh != ep.meta.nnzh), 1.0, 0.0))
    y = randn(ek.meta.ncon)
    push!(checks, Check("hess", "pert", "noelemhess host/port",
                        reldiff(hess_coord(ek, x, y), hess_coord(ep, x, y))...,
                        TOL_PORT))
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
    println(io, "problem\tdevice\tnvar\tncon\tbuild_host_s\tbuild_port_s\t" *
        "build_dev_s\tverdict\tcallback\tpoint\tcompare\trel\ttol\tmagnitude")
end
nfail = 0
for name in probs
    local checks, tb, tp, tg, nv, nc
    try
        checks, tb, tp, tg, nv, nc = verify(name)
        append!(checks, variants(name))
    catch e
        @printf("%-7s ERROR %s\n", name, sprint(showerror, e)[1:min(end, 300)])
        open(TSV, "a") do io
            println(io, "$name\t$DEVNAME\t?\t?\t?\t?\t?\tERROR\t-\t-\t-\t-\t-\t" *
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
        @printf("  %-14s %-5s %-22s %10.3e  tol %7.1e  mag %9.2e  %s\n",
            c.cb, c.pt, c.cmp, c.val, c.tol, c.mag,
            vacuous(c) ? "VACUOUS" : (ok(c) ? "ok" : "BAD"))
    end
    @printf("%-7s %-4s %d checks, %d bad, %d vacuous\n",
        name, verdict, length(checks), length(bad), length(vac))
    open(TSV, "a") do io
        for c in checks
            @printf(io, "%s\t%s\t%d\t%d\t%.1f\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3e\t%.1e\t%.3e\n",
                name, DEVNAME, nv, nc, tb, tp, tg,
                vacuous(c) ? "VACUOUS" : (ok(c) ? "ok" : "BAD"),
                c.cb, c.pt, c.cmp, c.val, c.tol, c.mag)
        end
    end
    flush(stdout)
    GC.gc()
end
if DEV === nothing
    println("\nCOLLOC_GPU_BACKEND=cpu: the gpu/host half was SKIPPED, not passed.")
end
@printf("\n%d/%d problems PASS\n", length(probs) - nfail, length(probs))
