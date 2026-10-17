# Compare every native (hand-written) model against the compiled Fortran it
# replaces, through `ExaCUTEstModel` rather than through each port's own
# constructor, so that the registry wiring is under test alongside the
# arithmetic: a port can be correct and still be reached by the wrong name,
# with the wrong default, or not at all.
#
# Run: julia --project=test test/native_verify.jl [PROBLEM ...]
# Results append to runs/native_verify.tsv as each problem finishes.
#
# WHAT A REFERENCE COMPARISON CAN AND CANNOT SETTLE. For most problems the
# compiled Fortran is right and equality with it is the whole claim. For nine
# of the 34 it is not, and those are named in `GRAD_MODE` / `HESS_MODE` below
# with the evidence for each. A derivative the reference does not supply or
# supplies wrongly is checked against a central difference of the lowest
# derivative that IS trustworthy -- the reference's own where possible, ours
# only where the reference's is the thing in question. Changing what a number
# is compared against is recorded; loosening a tolerance to cover a
# disagreement is not done anywhere in this file.
using ExaCUTEst, ExaModels, NLPModels, CUTEst
using LinearAlgebra, SparseArrays, Printf, Random
using FileWatching.Pidfile: mkpidlock

const LOCK = "/tmp/cutest-decode.pid"            # host-local: CUTEst compiles
const TSV = joinpath(@__DIR__, "..", "runs", "native_verify.tsv")

# Problems too large to assemble a Hessian for; `hprod` carries the claim
# there, in both directions of every random test direction.
const BIG = 50_000

# --- where the reference is not the standard ---------------------------------

# First derivatives -- gradient AND Jacobian together, because both are built
# from the same INDIVIDUALS `G` lines (or, for BLEACHNG, from the same
# sensitivity inhomogeneity), so a problem whose analytic first derivatives are
# wrong has both wrong. :direct unless listed.
#   :fd_ref -- central difference of the reference's own obj/cons, which ARE
#              correct; it is the analytic derivative of them that is not.
const FIRST_MODE = Dict(
    "HS67" => (:fd_ref, "the mastsif INDIVIDUALS G lines are inconsistent \
with the element's own F; fd of the reference's own values matches ours"),
    "HS85" => (:fd_ref, "two dropped minus signs in the mastsif G lines"),
    "BLEACHNG" => (:fd_ref, "ODESSA's DF omits the -y1/Vg term in the QIN \
column of the sensitivity inhomogeneity, so the reference's first derivatives \
are wrong there by 39-60%; fd of its own objective matches ours"),
)

# Hessian. :direct unless listed.
#   :fd_ref  -- central difference of the reference's Lagrangian gradient
#               (grad + J'y), used where the SIF supplies no H at all.
#   :fd_ours -- central difference of OUR Lagrangian gradient, used only where
#               the reference's gradient is itself wrong and so cannot serve.
const HESS_MODE = Dict(
    "CAR2" => (:fd_ref, "SIF Ktyp has F, A F and G lines and no H: \
classification LOR1 declares first derivatives only, and the reference \
Hessian is the SQR diagonal alone"),
    "ORBIT2" => (:fd_ref, "as CAR2: classification LOR1, no H lines"),
    "SARO" => (:fd_ref, "classification LOR1, no H section: the reference \
Hessian has the right structure and is identically zero"),
    "SAROMM" => (:fd_ref, "as SARO: classification LOR1, no H section"),
    "BLEACHNG" => (:fd_ours, "the reference Hessian is Gauss-Newton built on \
the broken QIN gradient column, so it reproduces that error rather than the \
curvature; our own gradient is validated against the reference's objective"),
    "HS67" => (:fd_ours, "the reference's H lines come from the same \
INDIVIDUALS block as its wrong G lines, so it cannot adjudicate; our own \
gradient is validated against the reference's values"),
    "HS85" => (:fd_ours, "as HS67: the H lines share the G lines' block"),
)

# Tolerances above the default, each with the mechanism that forces it. Keys
# are (problem, callback); callbacks are obj/grad/cons/jac/hess/hprod.
const TOL_DEFAULT = 1e-12
const TOL_FD = 1e-4          # central differences are truncation-limited
const TOL = Dict(
    ("CAR2", "cons") => (1e-8, "SIFdecode writes element parameters to \
OUTSDIF.d at 11 significant digits, and TOR = TN1 - TN cancels two numbers up \
to 1000x larger, so C2 = 3/(2 TOR) inherits ~9e-9; exact with \
ref_param_rounding = true"),
    ("CAR2", "jac") => (1e-8, "C2 is the largest Jacobian entry; same \
mechanism as cons"),
    ("ORBIT2", "cons") => (1e-8, "as CAR2, plus row L1's group constant \
RFRF = (RF/R0)^2, which the reference also stores truncated (2.7e-11)"),
    ("ORBIT2", "jac") => (1e-8, "as CAR2"),
    ("JIMACK", "obj") => (1e-11, "the reciprocal group SCALE reaches the \
Fortran as 5.7870370370D-05 against 1/17280 = 5.787037037037037e-5, a flat \
6.4e-12 on every value the group scales"),
    ("JIMACK", "grad") => (1e-11, "same group SCALE truncation"),
    ("JIMACK", "hprod") => (1e-11, "same group SCALE truncation"),
    ("JIMACK", "hess") => (1e-11, "same group SCALE truncation"),
    ("BLEACHNG", "obj") => (1e-8, "ODESSA integrates with RTOL = 1e-12 and \
the objective sums 1602 squared residuals; the port solves the same linear \
constant-coefficient ODE in closed form"),
    ("SARO", "jac") => (1e-14, "365 stages of RK4 summed in a different \
order than the Fortran's"),
    ("SAROMM", "jac") => (1e-14, "as SARO"),
    ("RAYBENDS", "hprod") => (1e-12, "1026 spline segments summed in a \
different order"),
    ("RAYBENDS", "hess") => (1e-12, "as hprod"),
)

tol(name, cb) = get(TOL, (name, cb), (TOL_DEFAULT, ""))[1]
why(name, cb) = get(TOL, (name, cb), (TOL_DEFAULT, ""))[2]

# --- helpers ----------------------------------------------------------------

# Relative infinity-norm difference. Returns 0.0 when both sides are zero, and
# `NaN` when either side is empty, so that a comparison which never had
# anything to compare cannot be read as agreement.
function rel(a, b)
    (isempty(a) || isempty(b)) && return NaN
    s = max(norm(a, Inf), norm(b, Inf))
    return s == 0 ? 0.0 : norm(a .- b, Inf) / s
end

# The reference reorders both variables and rows relative to SIF declaration
# order, so the names are what maps one indexing onto the other.
function refnames(ref)
    n = ref.meta.nvar; m = ref.meta.ncon
    st = zeros(Cint, 1)
    pname = Vector{Cchar}(undef, 10)
    vname = Matrix{Cchar}(undef, 10, n)
    g(M, k) = String(strip(join(Char(UInt8(c)) for c in M[:, k])))
    if m > 0
        cname = Matrix{Cchar}(undef, 10, m)
        CUTEst.cnames(Float64, ref.libsif, st, Cint[n], Cint[m], pname, vname, cname)
        return [g(vname, j) for j in 1:n], [g(cname, i) for i in 1:m]
    end
    CUTEst.varnames(Float64, ref.libsif, st, Cint[n], vname)
    return [g(vname, j) for j in 1:n], String[]
end

sparse_of(nlp, x, nr, nc) = sparse(jac_structure(nlp)..., jac_coord(nlp, x), nr, nc)
function hsparse(nlp, x, y)
    r, c = hess_structure(nlp)
    v = isempty(y) ? hess_coord(nlp, x) : hess_coord(nlp, x, y)
    # Duplicate coordinates are legitimate -- an element block repeats a
    # (row, col) pair that another block also touches -- and `sparse` sums
    # them, which is the accumulate semantics the oracle protocol specifies.
    n = nlp.meta.nvar
    return sparse(r, c, v, n, n)
end

# Lagrangian gradient of `nlp` at `x`, central-differenced along `v`.
function fd_hprod(nlp, x, y, v; h = nothing)
    hh = h === nothing ? 1e-6 * max(1.0, norm(x, Inf)) : h
    lg(z) = begin
        g = grad(nlp, z)
        isempty(y) ? g : g .+ sparse_of(nlp, z, nlp.meta.ncon, nlp.meta.nvar)' * y
    end
    return (lg(x .+ hh .* v) .- lg(x .- hh .* v)) ./ (2hh)
end

# Objective (and y-weighted constraint) value of `nlp`, central-differenced
# along `v`: the directional derivative of the Lagrangian.
function fd_dirderiv(nlp, x, y, v; h = nothing)
    hh = h === nothing ? 1e-6 * max(1.0, norm(x, Inf)) : h
    f(z) = obj(nlp, z) + (isempty(y) ? 0.0 : dot(y, cons(nlp, z)))
    return (f(x .+ hh .* v) - f(x .- hh .* v)) / (2hh)
end

# --- one problem ------------------------------------------------------------

struct Chk
    cb::String
    pt::String
    val::Float64
    tol::Float64
    mode::String
end
ok(c::Chk) = !isnan(c.val) && c.val <= c.tol

function verify(name; ndir = 3, seed = 20261016)
    Random.seed!(seed)
    checks = Chk[]
    tbuild = @elapsed ours = ExaCUTEstModel(name)
    nv, nc = ours.meta.nvar, ours.meta.ncon
    ovn = ExaCUTEst.sif_variable_names(name)
    ocn = nc > 0 ? ExaCUTEst.sif_constraint_names(name) : String[]
    gmode, gwhy = get(FIRST_MODE, name, (:direct, ""))
    hmode, hwhy = get(HESS_MODE, name, (:direct, ""))

    mkpidlock(LOCK) do
        ref = CUTEstModel(name)
        try
            rvn, rcn = refnames(ref)
            ref.meta.nvar == nv || error("nvar $(ref.meta.nvar) vs $nv")
            ref.meta.ncon == nc || error("ncon $(ref.meta.ncon) vs $nc")
            pv = Int[something(findfirst(==(s), ovn)) for s in rvn]
            pc = nc > 0 ? Int[something(findfirst(==(s), ocn)) for s in rcn] : Int[]
            length(unique(pv)) == nv || error("variable name map is not a permutation")

            toref(v) = v[pv]                       # our vector -> ref order
            fromref(v) = (w = similar(v); w[pv] = v; w)
            push!(checks, Chk("x0", "-", rel(toref(ours.meta.x0), ref.meta.x0),
                              1e-11, "SIFdecode writes x0 at ~12 digits"))
            # A bound vector's claim has two halves: the same entries are
            # infinite on both sides, and the finite ones agree. With no finite
            # entries the second half has nothing to compare -- which is the
            # usual case for an unbounded problem -- so it is recorded as
            # vacuous rather than passed, and the finiteness half still runs.
            function boundcheck(f, a, b)
                fin = isfinite.(b)
                if !all(isfinite.(a) .== fin)
                    push!(checks, Chk(f, "-", Inf, tol(name, f),
                                      "finiteness pattern differs"))
                elseif !any(fin)
                    push!(checks, Chk(f, "-", 0.0, tol(name, f),
                                      "vacuous: no finite entry on either side"))
                else
                    push!(checks, Chk(f, "-", rel(a[fin], b[fin]), tol(name, f),
                                      "$(count(fin)) finite of $(length(b))"))
                end
            end
            boundcheck("lvar", toref(ours.meta.lvar), ref.meta.lvar)
            boundcheck("uvar", toref(ours.meta.uvar), ref.meta.uvar)
            if nc > 0
                boundcheck("lcon", ours.meta.lcon[pc], ref.meta.lcon)
                boundcheck("ucon", ours.meta.ucon[pc], ref.meta.ucon)
            end

            x0r = ref.meta.x0
            for (pt, xr) in (("x0", x0r), ("pert", x0r .+ 0.01 .* randn(nv)))
                xo = fromref(xr)
                push!(checks, Chk("obj", pt, rel([obj(ours, xo)], [obj(ref, xr)]),
                                  tol(name, "obj"), ""))
                go = toref(grad(ours, xo))
                if gmode === :direct
                    push!(checks, Chk("grad", pt, rel(go, grad(ref, xr)),
                                      tol(name, "grad"), ""))
                else
                    # The reference's analytic gradient is wrong here; its
                    # values are not. Compare directional derivatives.
                    worst = 0.0
                    for _ in 1:ndir
                        vr = randn(nv); vr ./= norm(vr)
                        worst = max(worst, rel([dot(go, vr)],
                                               [fd_dirderiv(ref, xr, Float64[], vr)]))
                    end
                    push!(checks, Chk("grad", pt, worst, TOL_FD, "fd_ref: $gwhy"))
                end
                yr = Float64[]
                if nc > 0
                    push!(checks, Chk("cons", pt, rel(cons(ours, xo)[pc], cons(ref, xr)),
                                      tol(name, "cons"), ""))
                    Jo = sparse_of(ours, xo, nc, nv)[pc, pv]
                    if gmode === :direct
                        Jr = sparse_of(ref, xr, nc, nv)
                        push!(checks, Chk("jacstruct", pt,
                            Float64(nnz(dropzeros(Jo)) != nnz(dropzeros(Jr))), 0.0,
                            "nnz ours=$(nnz(dropzeros(Jo))) ref=$(nnz(dropzeros(Jr)))"))
                        push!(checks, Chk("jac", pt, rel(Jo, Jr), tol(name, "jac"), ""))
                    else
                        hh = 1e-6 * max(1.0, norm(xr, Inf))
                        worst = 0.0
                        for _ in 1:ndir
                            vr = randn(nv); vr ./= norm(vr)
                            worst = max(worst, rel(Jo * vr,
                                (cons(ref, xr .+ hh .* vr) .- cons(ref, xr .- hh .* vr)) ./ (2hh)))
                        end
                        push!(checks, Chk("jac", pt, worst, TOL_FD, "fd_ref: $gwhy"))
                    end
                    yr = randn(nc)
                end
                yo = nc > 0 ? (w = similar(yr); w[pc] = yr; w) : Float64[]

                worst = 0.0
                for _ in 1:ndir
                    vr = randn(nv); vr ./= norm(vr)
                    vo = fromref(vr)
                    hv = toref(nc > 0 ? hprod(ours, xo, yo, vo) : hprod(ours, xo, vo))
                    target = if hmode === :direct
                        nc > 0 ? hprod(ref, xr, yr, vr) : hprod(ref, xr, vr)
                    elseif hmode === :fd_ref
                        fd_hprod(ref, xr, yr, vr)
                    else
                        toref(fd_hprod(ours, xo, yo, vo))
                    end
                    worst = max(worst, rel(hv, target))
                end
                push!(checks, Chk("hprod", pt, worst,
                                  hmode === :direct ? tol(name, "hprod") : TOL_FD,
                                  hmode === :direct ? "" : "$hmode: $hwhy"))

                # Assembled Hessian: exercises hess_structure/hess_coord, which
                # hprod does not touch, and so is where a mis-declared sparsity
                # pattern or a non-accumulating write shows up.
                if nv <= BIG
                    Ho = hsparse(ours, xo, yo)
                    # `hess_coord!` fills one slot per declared nonzero. Where
                    # the declared pattern repeats a (row, col) pair -- element
                    # blocks that overlap -- the oracle protocol's accumulate
                    # semantics are what make the assembled matrix right, and
                    # the only reading that can tell an accumulating fill from
                    # an overwriting one is the matrix-vector product against
                    # `hprod`, which never touches the coordinate path. Both
                    # are ours, deliberately: this check is about the two
                    # paths agreeing, and the reference adjudicates separately
                    # below or through `hprod` above.
                    Hs = Ho .+ Ho' .- spdiagm(0 => diag(Ho))
                    worst = 0.0
                    for _ in 1:ndir
                        vo = randn(nv)
                        worst = max(worst, rel(Hs * vo,
                            nc > 0 ? hprod(ours, xo, yo, vo) : hprod(ours, xo, vo)))
                    end
                    push!(checks, Chk("hesscoord", pt, worst, 1e-12,
                        "assembled hess_coord against hprod, both ours: \
$(nnz(Ho)) declared, $(nnz(dropzeros(Ho))) distinct nonzeros"))
                    if hmode === :direct
                        Hr = hsparse(ref, xr, yr)
                        Hop = Ho[pv, pv]
                        push!(checks, Chk("hess", pt, rel(Hop, Hr), tol(name, "hess"), ""))
                        push!(checks, Chk("hessstruct", pt,
                            Float64(nnz(dropzeros(Hop)) != nnz(dropzeros(Hr))), 0.0,
                            "nnz ours=$(nnz(dropzeros(Hop))) ref=$(nnz(dropzeros(Hr)))"))
                    end
                end
            end
        finally
            finalize(ref)
        end
    end
    return checks, tbuild, nv, nc
end

# --- driver -----------------------------------------------------------------

probs = isempty(ARGS) ? ExaCUTEst.native_problems() : ARGS
mkpath(dirname(TSV))
isfile(TSV) || open(TSV, "w") do io
    println(io, "problem\tnvar\tncon\tbuild_s\tverdict\tcallback\tpoint\trel\ttol\tnote")
end
nfail = 0
for name in probs
    local checks, tb, nv, nc
    try
        checks, tb, nv, nc = verify(name)
    catch e
        @printf("%-11s ERROR %s\n", name, sprint(showerror, e)[1:min(end, 160)])
        open(TSV, "a") do io
            println(io, "$name\t?\t?\t?\tERROR\t-\t-\t-\t-\t$(sprint(showerror, e)[1:min(end,200)])")
        end
        global nfail += 1
        continue
    end
    bad = filter(!ok, checks)
    verdict = isempty(bad) ? "PASS" : "FAIL"
    isempty(bad) || (global nfail += 1)
    @printf("%-11s %-4s n=%-7d m=%-7d build %6.1fs  %d checks%s\n", name, verdict,
            nv, nc, tb, length(checks),
            isempty(bad) ? "" : "  >> " * join(("$(c.cb)/$(c.pt)=$(c.val)" for c in bad), " "))
    open(TSV, "a") do io
        for c in checks
            note = isempty(c.mode) ? why(name, c.cb) : c.mode
            @printf(io, "%s\t%d\t%d\t%.1f\t%s\t%s\t%s\t%.3e\t%.1e\t%s\n",
                    name, nv, nc, tb, ok(c) ? "ok" : "BAD", c.cb, c.pt, c.val, c.tol, note)
        end
    end
    flush(stdout)
end
@printf("\n%d/%d problems PASS\n", length(probs) - nfail, length(probs))
nfail == 0 || exit(1)
