# Compare SARO's and SAROMM's kernel-backed models on a GPU backend against the
# same kernels on the host, and both against `saro_model` / `saromm_model` --
# the host ports that `test/native_verify.jl` validates against the compiled
# Fortran. The chain is what makes a device result meaningful: a kernel can
# agree with its own CPU run and still integrate the wrong ODE.
#
# Run: julia --project=. test/saro_gpu_verify.jl [PROBLEM ...]
# Results append to runs/saro_gpu_verify.tsv as each problem finishes.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `SARO_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu`,
# which runs the host-versus-host half alone and is all that is available on a
# machine without a device. The `cpu` run records the device half as skipped,
# not as a pass.
#
# THE REFERENCE IS THE HOST PORT, NOT CUTEst, and here that is forced rather
# than convenient. Both SIFs supply no `H` section (classification `LOR1`,
# fourth character 1), so CUTEst's Hessian for these two problems is
# identically zero -- `hess_coord` all zeros and `hprod` the zero vector -- and
# cannot adjudicate a Hessian at all. The host port's own Hessian was
# validated against central differences of CUTEst's Jacobian (6e-11 .. 4e-10)
# and `test/native_verify.jl` is where that happens. What is tested here is
# that the kernels reproduce that port, which is a far tighter comparison.
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
#     reading, since it would catch a fill that writes a stage's values into
#     the wrong slots of its own block, which summing into a matrix can hide.
#     The patterns are compared too.
#  3. `hess_vs_hprod` -- `sparse(hess_structure, hess_coord) * v` against
#     `hprod`, inside one model. The only reading that exercises the COORDINATE
#     path: `hprod` goes through the oracle's `hvp!`, a separate pair of
#     kernels, and cannot see a mis-declared sparsity pattern or a fill that
#     writes the wrong slot. That is exactly how a duplicate-coordinate defect
#     survived the RAYBENDS and BA families' first verifications -- 96% and 92%
#     wrong assembled, exact in `hprod`.
#
# A FOURTH SECTION, `precisions`, records what the element types do, because
# for these two problems one of them is a known defect of the ported function
# rather than of the kernels: in Float32 the Jacobian and Hessian are
# non-finite at every stage (`src/native/SARO.jl` records the mechanism and the
# counts). The section asserts that the kernel builder reproduces the host
# port's non-finite COUNT rather than that it is finite, and that BigFloat --
# which is how a Float64 result gets arbitrated -- agrees with the port.
#
# Guarding against a vacuous pass: every check records the magnitude of what it
# compared, and a check whose two sides are both identically zero, or has an
# empty side, is recorded as `vacuous` and is NOT counted as agreement. SARO's
# objective at `x0` is one such: it is `X(7,KMAX)`, whose start value is zero.
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

# The kernel builders are loaded into the module here: `src/native.jl` does not
# include their file, so the package as installed has only the host ports.
isdefined(ExaCUTEst, :saro_kernel_model) || Base.include(
    ExaCUTEst, joinpath(@__DIR__, "..", "src", "native", "SARO_gpu.jl"))

const TSV = joinpath(@__DIR__, "..", "runs", "saro_gpu_verify.tsv")
const PROBS = ["SARO", "SAROMM"]

const KERNEL = Dict{String,Function}(
    "SARO" => (; kwargs...) -> ExaCUTEst.saro_kernel_model(; kwargs...),
    "SAROMM" => (; kwargs...) -> ExaCUTEst.saromm_kernel_model(; kwargs...),
)
const PORT = Dict{String,Function}(
    "SARO" => (; kwargs...) -> ExaCUTEst.saro_model(; kwargs...),
    "SAROMM" => (; kwargs...) -> ExaCUTEst.saromm_model(; kwargs...),
)

# --- tolerances, with the mechanism beside each -----------------------------

# Two host paths over the same 365 stages, calling the same
# `saro_stage_values` on the same Float32-rounded driving data. The kernels
# carry NARROWER derivative sweeps than the port -- one partial where the port
# uses eleven, two nested single partials where the port uses
# `ForwardDiff.hessian` -- and that changes nothing about the arithmetic,
# because a dual's partials are each computed from the same formulas and never
# read one another. The row values add their linear terms in the decode's
# order on both sides, and the Hessian's weighted sum is accumulated left to
# right on both sides.
#
# So `obj`, `grad`, `cons`, `jac` and `hess` are BIT-IDENTICAL -- measured 0 of
# 48180 Jacobian entries and 0 of 24090 Hessian entries differing, at both
# points on both problems -- and the bound is for the last bit of a value this
# port keeps bit-exact against the Fortran. Do not loosen it: a reassociation
# anywhere in the chain would show up here first.
const TOL_PORT = 1e-15

# `hprod` is the exception, and the mechanism is SARO's conditioning, not
# either implementation. `saro_oracle_model` supplies no `hvp!`, so its `hprod`
# accumulates its coordinate Hessian against `v`; `saro_kernel_model` supplies
# `hvp!`, which takes the second directional derivative along `v` in one AD
# sweep. The two are the SAME function in exact arithmetic -- built in BigFloat
# the two builders' `hess_coord` is bit-identical and their `hprod` agrees --
# so what separates them in Float64 is only where the rounding falls.
#
# And in Float64 one of SARO's stage Hessian entries carries real error of its
# own: at the perturbed point the `(U1, U1)` entry of stage 60 is 4.80988239
# where the same kernels in BigFloat give 4.809882390546975, so Float64 loses
# about five digits there. Measured against that BigFloat value, the HOST is
# 1.28e-11 away and the DEVICE 1.25e-11 -- the device is the marginally more
# accurate of the two. The gap between any two Float64 readings of that entry
# is therefore ~2.5e-11 relative to the model's `|H|max`, and that is a
# property of the function at that point, not evidence about either path.
# `U1` enters through `v6 = 1000 U1 (0.001 PSH) / (69666.67 - 1161.11 TRSH)`
# and then the regularised quotient, and the second derivative cancels.
const TOL_HPROD = 1e-9

# Device against the same kernels on the host, BY CALLBACK. A stage is eight
# `saro_rhs` evaluations of ~60 operations, and the device's FMA contraction is
# not the host's; nothing is summed across stages except in `hprod`, where the
# gather table fixes the order on both backends, so these are per-stage
# rounding bounds and not reduction ones.
#
# The first-order callbacks are far inside their bound -- measured worst 2.1e-15
# on `cons` and 4.3e-16 on `jac`, both problems, both points -- and are held at
# 1e-13 because a device fault in a kernel like this one would not be subtle: a
# wrong launch partition or a wrong slot moves an entry by O(1), not by an ulp.
#
# Second order gets a looser bound at the perturbed point for the reason
# `TOL_HPROD` records, and only there: exactly one of SARO's 24090 Hessian
# entries exceeds 1e-11 (two exceed 1e-13), it is the ill-conditioned
# `(U1, U1)` entry of stage 60, and arbitration against BigFloat puts the
# device nearer the exact value than the host. At `x0` the worst second-order
# reading is 6.2e-14.
tol_dev(cb, pt) =
    cb in ("hess", "hprod") ? (pt == "pert" ? 1e-10 : 1e-12) : 1e-13

# Second order against first order inside one model: the same AD Hessian read
# once through the coordinate path and once through the matrix-free one, which
# is `TOL_HPROD`'s mechanism again, so it carries `TOL_HPROD`'s bound.
const TOL_SELF = TOL_HPROD

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "SARO_GPU_BACKEND", "cuda")
if WANT == "cuda"
    @eval using CUDA
end
const DEV = WANT == "cuda" ? CUDA.CUDABackend() : nothing
const DEVNAME = WANT == "cuda" ? "cuda" : "cpu"
if WANT == "cuda"
    CUDA.allowscalar(false)
    @info "device" name = CUDA.name(CUDA.device())
end

# BigFloat `hess` is 24090 stage evaluations under nested BigFloat duals, so it
# is opt-in; `cons` and `jac` in BigFloat run unconditionally.
const WITH_BIG_HESS = get(ENV, "SARO_GPU_BIG_HESS", "0") == "1"

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

# Count of entries that differ in the last bit, for the quantities this port
# keeps bit-exact. Reported beside the relative difference because `0.0` can
# also mean "both sides were zero", which `mag` then discloses.
bitdiff(a, b) = (a = Array(a); b = Array(b);
                 length(a) == length(b) ? count(a .!= b) : -1)

# A lower triangle assembled into a symmetric matvec, so a Hessian can be
# checked against the hprod of the same model without either side's duplicate
# coordinate convention entering. `sparse` sums duplicates, which is the
# convention a blockwise pattern relies on.
function hess_times(n, rows, cols, vals, v)
    A = sparse(Array(rows), Array(cols), Array(vals), n, n)
    return A * v .+ (A - Diagonal(A))' * v
end

# A model's declared Hessian pattern, brought to the host. `similar(x, ...)`
# puts the arrays wherever the model lives, so `hess_structure!` never has to
# index a device array from here -- which `hess_structure(nlp)` would.
function hess_pattern(nlp, x)
    r = similar(x, Int, nlp.meta.nnzh)
    c = similar(x, Int, nlp.meta.nnzh)
    hess_structure!(nlp, r, c)
    return Array(r), Array(c)
end

function jac_pattern(nlp, x)
    r = similar(x, Int, nlp.meta.nnzj)
    c = similar(x, Int, nlp.meta.nnzj)
    jac_structure!(nlp, r, c)
    return Array(r), Array(c)
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

# As `Check`, for a count that must match exactly -- a pattern, or the number
# of non-finite Float32 entries. `mag = 1.0` keeps a match from reading as
# vacuous: what is compared is never empty.
exact(cb, pt, cmp, differs) = Check(cb, pt, cmp, Float64(differs), 1.0, 0.0)

function verify(name)
    checks = Check[]
    tb = @elapsed host = KERNEL[name](; backend = nothing)
    tp = @elapsed port = PORT[name]()
    tg = DEV === nothing ? 0.0 : @elapsed KERNEL[name](; backend = DEV)
    gpu = DEV === nothing ? nothing : ExaCUTEstModel(name; backend = DEV)
    n, m = host.meta.nvar, host.meta.ncon
    (n == port.meta.nvar && m == port.meta.ncon) ||
        error("$name: kernel model is $n x $m, port is " *
              "$(port.meta.nvar) x $(port.meta.ncon)")
    @printf("%-7s nvar=%d ncon=%d  nnzj %d/%d  nnzh %d/%d  build %.1f/%.1f/%.1f s\n",
        name, n, m, host.meta.nnzj, port.meta.nnzj, host.meta.nnzh,
        port.meta.nnzh, tb, tp, tg)

    x0h = Array(host.meta.x0)
    jrh, jch = jac_pattern(host, x0h)
    hrh, hch = hess_pattern(host, x0h)
    jrp, jcp = jac_pattern(port, Array(port.meta.x0))
    hrp, hcp = hess_pattern(port, Array(port.meta.x0))
    push!(checks, exact("jac_structure", "-", "host/port",
                        jrh != jrp || jch != jcp))
    push!(checks, exact("hess_structure", "-", "host/port",
                        hrh != hrp || hch != hcp))

    Random.seed!(11)
    for (pt, xh) in (("x0", copy(x0h)), ("pert", x0h .+ 0.01 .* randn(n)))
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
        cp = cons(port, xh)
        jp = jac_coord(port, xh)
        hp = hess_coord(port, xh, yh)
        push!(checks, Check("cons", pt, "host/port", reldiff(ch, cp)..., TOL_PORT))
        push!(checks, Check("jac", pt, "host/port", reldiff(jh, jp)..., TOL_PORT))
        push!(checks, Check("hess", pt, "host/port", reldiff(hh, hp)..., TOL_PORT))
        push!(checks, Check("hprod", pt, "host/port",
                            reldiff(ph, hprod(port, xh, yh, vh))..., TOL_HPROD))
        # Bit-exactness, asserted rather than inferred from a 0.0 relative
        # difference: these three are what ties the kernels to the Fortran.
        push!(checks, exact("cons_bits", pt, "host/port", bitdiff(ch, cp) != 0))
        push!(checks, exact("jac_bits", pt, "host/port", bitdiff(jh, jp) != 0))
        push!(checks, exact("hess_bits", pt, "host/port", bitdiff(hh, hp) != 0))

        # --- 3. the coordinate Hessian against the matrix-free one
        push!(checks, Check("hess_vs_hprod", pt, "host",
                            reldiff(hess_times(n, hrh, hch, hh, vh), ph)...,
                            TOL_SELF))

        # --- 1. the device against the same kernels on the CPU backend
        gpu === nothing && continue
        push!(checks, Check("obj", pt, "gpu/host", reldiff(fh, obj(gpu, xd))...,
                            tol_dev("obj", pt)))
        push!(checks, Check("grad", pt, "gpu/host", reldiff(gh, grad(gpu, xd))...,
                            tol_dev("grad", pt)))
        push!(checks, Check("cons", pt, "gpu/host", reldiff(ch, cons(gpu, xd))...,
                            tol_dev("cons", pt)))
        push!(checks, Check("jac", pt, "gpu/host",
                            reldiff(jh, jac_coord(gpu, xd))..., tol_dev("jac", pt)))
        hd = hess_coord(gpu, xd, yd)
        pd = hprod(gpu, xd, yd, vd)
        push!(checks, Check("hess", pt, "gpu/host", reldiff(hh, hd)...,
                            tol_dev("hess", pt)))
        push!(checks, Check("hprod", pt, "gpu/host", reldiff(ph, pd)...,
                            tol_dev("hprod", pt)))

        # The sparsity patterns are built on the host and shipped, so a
        # disagreement here is a transfer fault rather than an arithmetic one.
        rr, cc = hess_pattern(gpu, xd)
        push!(checks, exact("hess_structure", pt, "gpu/host",
                            rr != hrh || cc != hch))
        jr, jc = jac_pattern(gpu, xd)
        push!(checks, exact("jac_structure", pt, "gpu/host",
                            jr != jrh || jc != jch))

        # ...and comparison 3 on the device, which is where a coordinate fill
        # that only works on one thread per item would show.
        push!(checks, Check("hess_vs_hprod", pt, "gpu",
                            reldiff(hess_times(n, rr, cc, hd, vh), Array(pd))...,
                            TOL_SELF))
    end
    return checks, tb, tp, tg, n, m
end

# The element types, on the host alone: what is at stake is a property of the
# ported function, not of the hardware. No kernel here uses
# `KernelAbstractions.@atomic`, so the kernel builder is expected to serve
# BigFloat with no gate -- which is the claim this section tests.
function precisions(name)
    checks = Check[]
    Random.seed!(11)

    k32 = KERNEL[name](; T = Float32)
    p32 = PORT[name](; T = Float32)
    n, m = k32.meta.nvar, k32.meta.ncon
    x32 = Array(k32.meta.x0)
    y32 = Float32.(randn(m))
    # `cons` IS finite in Float32 and must agree with the port; the derivatives
    # are not, and what is asserted of them is that the kernel builder produces
    # the SAME non-finite entries as the port, since a different count would
    # mean a second defect on top of the known one.
    ck, cp = cons(k32, x32), cons(p32, x32)
    push!(checks, Check("cons", "x0", "f32 host/port", reldiff(ck, cp)..., 1e-7))
    push!(checks, exact("cons_finite", "x0", "f32 kernel",
                        count(!isfinite, Array(ck)) != 0))
    for (cb, a, b) in (("jac", jac_coord(k32, x32), jac_coord(p32, x32)),
                       ("hess", hess_coord(k32, x32, y32),
                        hess_coord(p32, x32, y32)))
        na = count(!isfinite, Array(a))
        nb = count(!isfinite, Array(b))
        push!(checks, exact("$(cb)_nonfinite", "x0", "f32 host/port", na != nb))
        @printf("  %-7s Float32 non-finite: kernel %d, port %d, of %d\n",
                cb, na, nb, length(Array(a)))
    end

    kB = KERNEL[name](; T = BigFloat)
    pB = PORT[name](; T = BigFloat)
    xB = Array(kB.meta.x0)
    yB = BigFloat.(randn(m))
    push!(checks, Check("cons", "x0", "big host/port",
                        reldiff(cons(kB, xB), cons(pB, xB))..., TOL_PORT))
    push!(checks, Check("jac", "x0", "big host/port",
                        reldiff(jac_coord(kB, xB), jac_coord(pB, xB))..., TOL_PORT))
    if WITH_BIG_HESS
        push!(checks, Check("hess", "x0", "big host/port",
                            reldiff(hess_coord(kB, xB, yB),
                                    hess_coord(pB, xB, yB))..., TOL_PORT))
    end
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
        append!(checks, precisions(name))
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
        @printf("  %-16s %-5s %-18s %10.3e  tol %7.1e  mag %9.2e  %s\n",
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
    println("\nSARO_GPU_BACKEND=cpu: the gpu/host half was SKIPPED, not passed.")
end
@printf("\n%d/%d problems PASS\n", length(probs) - nfail, length(probs))
