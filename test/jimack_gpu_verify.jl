# Compare JIMACK's kernel-backed model on a GPU backend against the same model
# on the host, and both against `jimack_model` -- the host port that
# `test/native_verify.jl` validates against the compiled Fortran. The chain is
# what makes a device result meaningful: a kernel can agree with its own CPU
# run and still compute the wrong energy.
#
# Run: julia --project=. test/jimack_gpu_verify.jl
# Results append to runs/jimack_gpu_verify.tsv.
#
# SCALAR INDEXING IS DISABLED FOR THE WHOLE RUN (`CUDA.allowscalar(false)`) and
# never re-enabled. That is half the claim being tested: a model that evaluates
# only because the host is allowed to touch device memory element by element is
# not on the device in any useful sense. The other half is agreement.
#
# `JIMACK_GPU_BACKEND` selects the device backend: `cuda` (default) or `cpu` to
# compare the two host paths alone, which is all that is available on a machine
# without a GPU. Under `cpu` the `gpu/host` rows compare the host kernels with
# themselves and are therefore vacuous by construction -- they read exactly
# 0.0 -- and only the `host/port` and `hess_vs_hprod` rows carry information.
#
# `JIMACK_GPU_CUTEST=1` adds the reference `CUTEstModel` as an independent
# anchor; it needs gfortran and the shared decode directory.
#
# THE THREE COMPARISONS, and why none of them subsumes another.
#
#  1. `gpu/host` -- the device against the SAME kernels on KernelAbstractions'
#     CPU backend, so a difference is the device and not a different algorithm.
#     Both declare the same Hessian pattern, so `hess` compares the value
#     array slot by slot and `hess_asm` the assembled matrix.
#
#  2. `host/port` -- the host kernels against `jimack_model`. That is the one
#     that reaches the Fortran-verified reference, so it is the one that
#     matters most. The two models declare DIFFERENT Hessian patterns
#     (`jimack_kernel_model` is blockwise with duplicates, `jimack_model` is
#     the deduplicated union), so their Hessians are compared as ASSEMBLED
#     matrices, entry by entry. That is pattern-independent and would catch a
#     mis-declared pattern on either side.
#
#  3. `hess_vs_hprod` -- the assembled coordinate Hessian against `hprod`,
#     within one model. This is the only reading that exercises the coordinate
#     path at all: `hprod` goes through `hvp!` and never touches a declared
#     coordinate, so it cannot see a mis-declared pattern or a fill that
#     writes the wrong slot. That is exactly how a duplicate-coordinate defect
#     survived a first verification on a BA objective -- 92% wrong assembled,
#     exact in `hprod`.
include(joinpath(@__DIR__, "gpu", "activate.jl"))
using ExaCUTEst, ExaModels, NLPModels
using LinearAlgebra, SparseArrays, Printf, Random

const TSV = joinpath(@__DIR__, "..", "runs", "jimack_gpu_verify.tsv")

# --- tolerances, with the mechanism beside each -----------------------------

# Two host paths over the same 864 elements. The element values themselves are
# bit-identical -- both paths call `jimack_element_value` on the same
# precomputed geometry -- but the kernel model reduces them with `sum` while
# `jimack_model` accumulates sequentially, and the gradient/hvp scatters run in
# different orders.
const TOL_HOST = 1e-12

# Device against host kernels, BY POINT. The gap between the two bounds is a
# property of JIMACK at the perturbed point, not of the device.
#
# At the undeformed mesh the two sides differ only by the device's `log` and
# its FMA contraction, an ulp apiece, and nothing amplifies it: both sit
# 1.4e-14 from the same kernels evaluated in BigFloat.
#
# The perturbed point displaces every node by ~0.01 while the cells are
# `DELTA/M = 0.0167` tall, so the deformation gradient is far from the
# identity and `INV3`, a sum of cancelling triple products, loses most of its
# significance; second derivatives of `log(INV3)` inherit that. The largest
# element Hessian block there is 7.9e12, and arbitrating both sides against
# BigFloat puts the HOST 1.5e-9 away from it and the DEVICE 5.3e-11 away --
# the device is the MORE accurate of the two, because fusing the multiply-adds
# of `INV3` cancels less. So this bound is the host's Float64 noise floor at
# that point, and a device result inside it is not evidence of a device defect.
#
# `hess_coord` is held to the same bound as the scatters even though it has no
# scatter of its own -- each element owns a disjoint 300-slot slice -- because
# the mechanism is the conditioning, not the summation order.
tol_dev(pt) = pt == "x0" ? 1e-13 : 1e-8

# The assembled coordinate Hessian times v against `hvp!`, within one model.
# Both sum 864 element blocks in different orders, over entries as large as
# 5e8 at the perturbed point.
const TOL_CANCEL = 1e-11

# Against CUTEst. SIFdecode writes the reciprocal group SCALE into OUTSDIF.d as
# `5.7870370370D-05` where 1/17280 = 5.787037037037037e-5, so every CUTEst
# quantity for this problem is uniformly (1 - 6.4e-12) times the exactly
# scaled one. This port divides by 17280, which is correct; the offset is
# CUTEst's text round-trip and is not adopted. At the perturbed point the
# Float64 noise floor described above is larger than that offset and is what
# bounds the comparison.
tol_ref(pt) = max(1e-11, tol_dev(pt))

# --- backend ----------------------------------------------------------------

const WANT = get(ENV, "JIMACK_GPU_BACKEND", "cuda")
if WANT == "cuda"
    @eval using CUDA
end
const DEV = WANT == "cuda" ? CUDA.CUDABackend() : nothing
const DEVNAME = WANT == "cuda" ? "cuda" : "cpu"
if WANT == "cuda"
    CUDA.allowscalar(false)
    @info "device" name = CUDA.name(CUDA.device())
end

const WITH_REF = get(ENV, "JIMACK_GPU_CUTEST", "0") == "1"
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
isdefined(ExaCUTEst, :jimack_kernel_model) || Base.include(
    ExaCUTEst, joinpath(@__DIR__, "..", "src", "native", "JIMACK_gpu.jl"))

# The registry still routes JIMACK to the host-loop builder, so the device
# model is pointed at the kernel builder here. Reaching it through
# `ExaCUTEstModel` is deliberate: it puts the registry wiring and the keyword
# forwarding under test alongside the arithmetic.
ExaCUTEst.NATIVE_MODELS["JIMACK"] =
    (; kwargs...) -> ExaCUTEst.jimack_kernel_model(; kwargs...)

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

# As `rel`, over two sparse matrices. `norm(A, Inf)` of a matrix is the
# entrywise maximum, so this never materializes a dense 3549x3549 array.
function relmat(A, B)
    (nnz(A) == 0 || nnz(B) == 0) && return NaN
    s = max(norm(A, Inf), norm(B, Inf))
    return s == 0 ? 0.0 : norm(A - B, Inf) / s
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

# Those coordinates as a sparse lower triangle. Duplicates are summed by
# `sparse`, which is the convention the declared pattern relies on.
hess_matrix(n, (r, c, v)) = sparse(r, c, v, n, n)

# A lower triangle assembled into a symmetric matvec, so a Hessian can be
# checked against the hprod of the same model without either side's sparsity
# pattern or duplicate-coordinate convention entering.
hess_times(A, v) = A * v .+ (A - Diagonal(A))' * v

# The same triangle as a full symmetric matrix, which is what survives a
# variable permutation: permuting a lower triangle does not keep it lower
# triangular.
symmetrize(A) = A + (A - Diagonal(A))'

# CUTEst's variable names, in CUTEst's own order.
function refvarnames(ref)
    n = ref.meta.nvar
    st = zeros(Cint, 1)
    pname = Vector{Cchar}(undef, 10)
    vname = Matrix{Cchar}(undef, 10, n)
    CUTEst.varnames(Float64, ref.libsif, st, Cint[n], vname)
    return [String(strip(join(Char(UInt8(c)) for c in vname[:, j]))) for j in 1:n]
end

struct Check
    cb::String
    pt::String
    cmp::String
    val::Float64
    tol::Float64
end
ok(c) = isfinite(c.val) && c.val <= c.tol

function verify()
    checks = Check[]
    th = @elapsed host = ExaCUTEst.jimack_kernel_model()
    tp = @elapsed port = ExaCUTEst.jimack_model()
    tg = @elapsed gpu = ExaCUTEstModel("JIMACK"; backend = DEV)
    n = host.meta.nvar
    @printf("nvar=%d  nnzh kernel=%d  nnzh port=%d  build %.1f/%.1f/%.1f s\n",
        n, host.meta.nnzh, port.meta.nnzh, th, tp, tg)

    Random.seed!(11)
    x0 = Array(host.meta.x0)

    for (pt, xh) in (("x0", copy(x0)), ("pert", x0 .+ 0.01 .* randn(n)))
        vh = randn(n)
        xd, vd = dev(xh), dev(vh)

        td = tol_dev(pt)
        fh = obj(host, xh)
        gh = grad(host, xh)
        ph = hprod(host, xh, vh)
        ch = hess_coords(host, xh)
        Ah = hess_matrix(n, ch)

        # (2) host kernels against the Fortran-verified host port. The two
        # declare different Hessian patterns, so that comparison is on the
        # assembled matrices.
        push!(checks, Check("obj", pt, "host/port", relsc(fh, obj(port, xh)), TOL_HOST))
        push!(checks, Check("grad", pt, "host/port", rel(gh, grad(port, xh)), TOL_HOST))
        push!(checks, Check("hprod", pt, "host/port",
            rel(ph, hprod(port, xh, vh)), TOL_HOST))
        push!(checks, Check("hess_asm", pt, "host/port",
            relmat(Ah, hess_matrix(n, hess_coords(port, xh))), TOL_HOST))

        # (1) the device against the same kernels on the host.
        push!(checks, Check("obj", pt, "gpu/host", relsc(fh, obj(gpu, xd)), td))
        push!(checks, Check("grad", pt, "gpu/host", rel(gh, grad(gpu, xd)), td))
        push!(checks, Check("hprod", pt, "gpu/host",
            rel(ph, hprod(gpu, xd, vd)), td))

        # Here the two patterns ARE the same, so the Hessian is compared slot
        # by slot as well as assembled. Slot by slot is the stronger of the
        # two: given equal structure it would catch a fill that writes an
        # element's values into the wrong slots of its own block, which summing
        # into a matrix can hide.
        cg = hess_coords(gpu, xd)
        push!(checks, Check("hess_structure", pt, "gpu/host",
            Float64(cg[1] != ch[1] || cg[2] != ch[2]), 0.0))
        push!(checks, Check("hess", pt, "gpu/host", rel(ch[3], cg[3]), td))
        Ag = hess_matrix(n, cg)
        push!(checks, Check("hess_asm", pt, "gpu/host", relmat(Ah, Ag), td))

        # (3) the coordinate path against first-order information, in each
        # model under test.
        push!(checks, Check("hess_vs_hprod", pt, "host",
            rel(hess_times(Ah, vh), ph), TOL_CANCEL))
        push!(checks, Check("hess_vs_hprod", pt, "gpu",
            rel(hess_times(Ag, vh), Array(hprod(gpu, xd, vd))), TOL_CANCEL))

        if WITH_REF
            mkpidlock(LOCK) do
                ref = CUTEstModel("JIMACK")
                try
                    # CUTEst reorders variables; map by SIF name, with a Dict
                    # rather than a scan per name.
                    ivn = Dict(s => j
                               for (j, s) in enumerate(ExaCUTEst.sif_variable_names("JIMACK")))
                    pv = Int[ivn[s] for s in refvarnames(ref)]
                    length(unique(pv)) == n ||
                        error("variable name map is not a permutation")
                    xr = xh[pv]
                    vr = vh[pv]
                    push!(checks, Check("obj", pt, "host/cutest",
                        relsc(fh, obj(ref, xr)), tol_ref(pt)))
                    push!(checks, Check("grad", pt, "host/cutest",
                        rel(gh[pv], grad(ref, xr)), tol_ref(pt)))
                    push!(checks, Check("hprod", pt, "host/cutest",
                        rel(ph[pv], hprod(ref, xr, vr)), tol_ref(pt)))
                    # The coordinate path against the reference's own, entry by
                    # entry: the check `native_verify.jl` cannot afford for a
                    # port whose `ScalarNonlinearOracle` has no explicit
                    # `hess!`, since reconstructing one costs `nvar` products.
                    Ar = let (rr, rc) = hess_structure(ref)
                        sparse(rr, rc, hess_coord(ref, xr), n, n)
                    end
                    Sh = symmetrize(Ah)[pv, pv]
                    push!(checks, Check("hess", pt, "host/cutest",
                        relmat(Sh, symmetrize(Ar)), tol_ref(pt)))
                finally
                    finalize(ref)
                end
            end
        end
    end
    return checks, n, th, tp, tg
end

# --- driver -----------------------------------------------------------------

mkpath(dirname(TSV))
isfile(TSV) || open(TSV, "w") do io
    println(io, "problem\tdevice\tnvar\tbuild_host_s\tbuild_port_s\tbuild_dev_s\t" *
        "verdict\tcallback\tpoint\tcompare\trel\ttol")
end

checks, n, th, tp, tg = verify()
bad = filter(!ok, checks)
verdict = isempty(bad) ? "PASS" : "FAIL"
open(TSV, "a") do io
    for c in checks
        @printf(io, "JIMACK\t%s\t%d\t%.1f\t%.1f\t%.1f\t%s\t%s\t%s\t%s\t%.3e\t%.1e\n",
            DEVNAME, n, th, tp, tg, ok(c) ? "ok" : "BAD", c.cb, c.pt, c.cmp,
            c.val, c.tol)
    end
end
for c in checks
    @printf("%-14s %-5s %-12s %10.3e  tol %7.1e  %s\n",
        c.cb, c.pt, c.cmp, c.val, c.tol, ok(c) ? "ok" : "BAD")
end
@printf("\nJIMACK %s on %s: %d checks, %d bad\n",
    verdict, DEVNAME, length(checks), length(bad))
