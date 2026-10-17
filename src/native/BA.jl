# Bundle adjustment: a generic Julia port of the BA-L* external Fortran.
#
# Written as a plain, type-generic Julia function -- NOT as an ExaModels
# expression (Sungho 2026-10-16: "it should be a generic julia function").
# That matters for three reasons:
#   * it compiles like ordinary Julia code. Building the same computation as a
#     symbolic expression inside a SIF element did not finish compiling in an
#     hour on a 57-variable problem, because ExaModels nests a type parameter
#     per operation and this chain is ~30 deep;
#   * being generic in T, it evaluates on Float64 for verification, on Dual
#     numbers for AD derivatives, and on BigFloat for a precision check, with
#     no separate code paths;
#   * it contains no allocation, no branching on values and no indexing, so it
#     is callable from a GPU kernel -- the "scalar indexing disabled"
#     requirement of roadmap step 4.
#
# The Fortran's hand-coded G(12) and H(78) derivative arrays are NOT ported;
# derivatives come from AD over this one function. That is what keeps the port
# clear of the error class that made the legacy HS67 port wrong in the 4th
# digit.
#
# THE ROTATION DISCREPANCY. The Fortran's second camera coordinate reads
#     cy = x2*ct + (ez*x1 - ex*x3)*st + ex*p
# where Rodrigues requires `ey*p` in the last term; the first and third
# coordinates match the formula exactly, and ex/ey/ez are each assigned once
# and never reassigned. Measured on BA-L1 at x0, the as-written form
# reproduces CUTEst's cons(x0) to all 16 digits (-9.020226512912586,
# 11.194619796465815) and the textbook form does not (-9.02022648716553,
# 11.263958473662967). So the shipped problems do not compute the Snavely
# error their own header documents. Both forms are available; `rodrigues`
# selects, defaulting to the Fortran so a port verifies against the reference.

"""
    ba_residual(rx, ry, rz, px, py, pz, tx, ty, tz, k1, k2, fl, yres;
                rodrigues = false)

The projected pixel coordinate of one bundle-adjustment observation: Rodrigues
rotation of the point, translation, perspective division, radial distortion
`1 + k1 r^2 + k2 r^4`, focal scaling. `yres` selects the component -- 0 for x,
1 for y -- and is a blend coefficient rather than a branch, so the function
stays straight-line code.

The observed coordinate is NOT subtracted here: in the SIF it is the row's
group constant.

Generic in the element type, allocation-free, branch-free on values.
"""
@inline function ba_residual(rx::T, ry::T, rz::T, px::T, py::T, pz::T,
                             tx::T, ty::T, tz::T, k1::T, k2::T, fl::T,
                             yres::T; rodrigues::Bool = false) where {T}
    tt = rx * rx + ry * ry + rz * rz
    t = sqrt(tt)
    ct, st = cos(t), sin(t)
    it = inv(t)
    ex, ey, ez = rx * it, ry * it, rz * it
    pp = (ex * px + ey * py + ez * pz) * (one(T) - ct)

    cx = px * ct + (ey * pz - ez * py) * st + ex * pp
    cy = py * ct + (ez * px - ex * pz) * st + (rodrigues ? ey : ex) * pp
    cz = pz * ct + (ex * py - ey * px) * st + ez * pp

    cx += tx
    cy += ty
    cz += tz

    icz = inv(cz)
    dx = -cx * icz
    dy = -cy * icz
    cc = dx * dx + dy * dy
    d = one(T) + k1 * cc + k2 * cc * cc
    return fl * d * (dx + yres * (dy - dx))
end

"Twelve elemental variables as a tuple, in the SIF's declaration order."
@inline ba_residual(v::NTuple{12,T}, yres::T; rodrigues::Bool = false) where {T} =
    ba_residual(v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9],
                v[10], v[11], v[12], yres; rodrigues)

# ---------------------------------------------------------------------------
# Interfacing the generic function to ExaModels as an external oracle
# (Sungho 2026-10-16: "this generic julia function can be interfaced with
# ExaModels as an external oracle").
#
# Everything below is assembly: the mathematics lives entirely in
# ba_residual, and every derivative comes from AD over it, so the Fortran's
# hand-coded G(12)/H(78) arrays are never ported. Each observation touches
# exactly 12 variables, which makes the sparsity exact and local: 12 Jacobian
# entries per row, and a 78-entry lower triangle per observation in the
# Hessian.

"The twelve elemental variables, in the SIF's declaration order."
const BA_EVARS = (:RX, :RY, :RZ, :X, :Y, :Z, :TX, :TY, :TZ, :KA, :KB, :F)

"One observation: where its 12 variables live, which component, and its row."
# Parametric in the model's element type: with these three fields fixed at
# Float64, `ba_row` returns a Float64 whatever `x` is, and the objective's
# accumulator promotes after its first term -- so a `T = Float32` model sums
# in Float64 and rounds once at the end. The result is numerically no worse
# (measured better than Float64's own error against a BigFloat sum), but it is
# a type instability in the hot loop and the arithmetic is not the precision
# the model was asked for.
struct BAObs{T}
    idx::NTuple{12,Int}
    yres::T
    w::T
    b::T
end

"""
    ba_observations(P, T = Float64) -> (Vector{BAObs{T}}, rows, isobj)

Read the observations out of a `sif_decode(...; external = :defer)` problem:
each row holds exactly one element of type `E`, whose twelve elemental
variables map to problem variables, with `YRES` selecting the component and
the observed pixel arriving as the row's group constant.
"""
function ba_observations(P, ::Type{T} = Float64) where {T}
    rows = [g for g in P.grouporder if P.groups[g].kind != 'N']
    isobj = isempty(rows)
    isobj && (rows = copy(P.grouporder))
    obs = Vector{BAObs{T}}(undef, length(rows))
    for (i, g) in enumerate(rows)
        grp = P.groups[g]
        length(grp.elems) == 1 ||
            error("$g holds $(length(grp.elems)) elements; expected exactly 1")
        en, w = grp.elems[1]
        e = P.elements[en]
        obs[i] = BAObs{T}(ntuple(k -> e.varmap[BA_EVARS[k]], 12),
                          e.parvals[:YRES], w, grp.constant)
    end
    return obs, rows, isobj
end


@inline _ba_gather(x, o::BAObs) = ntuple(k -> x[o.idx[k]], 12)

"Row value: the port's projection, weighted, with the observation folded in."
@inline function ba_row(x, o::BAObs; rodrigues::Bool = false)
    return o.w * ba_residual(_ba_gather(x, o), oftype(first(x), o.yres); rodrigues) - o.b
end

"Row gradient with respect to its 12 variables, by AD over ba_residual."
function ba_row_grad(x, o::BAObs; rodrigues::Bool = false)
    v = collect(_ba_gather(x, o))
    g = ForwardDiff.gradient(
        z -> ba_residual(NTuple{12,eltype(z)}(z), convert(eltype(z), o.yres); rodrigues), v)
    return o.w .* g
end

"Row Hessian (12x12) with respect to its 12 variables, by AD over ba_residual."
function ba_row_hess(x, o::BAObs; rodrigues::Bool = false)
    v = collect(_ba_gather(x, o))
    H = ForwardDiff.hessian(
        z -> ba_residual(NTuple{12,eltype(z)}(z), convert(eltype(z), o.yres); rodrigues), v)
    return o.w .* H
end

"""
    ba_oracle_model(name; T = Float64, rodrigues = false) -> ExaModel

Build a BA-L* problem as an ExaModel whose callbacks are the generic Julia
port rather than compiled Fortran.

The constrained variants (`BA-L1`, `BA-L16`, ...) have no objective and one
equality row per observation, so they become a `VectorNonlinearOracle`. The
`*LS` variants have no constraints and a sum-of-squared-residuals objective,
so they become a `ScalarNonlinearOracle`.

Needs no Fortran toolchain and no `sifdecoder`: the data comes from
`sif_decode(...; external = :defer)` and the functions from `ba_residual`.
That also makes `BA-L52` cheap to construct, where CUTEst spends ~1035 s
compiling it.
"""
function ba_oracle_model(name::AbstractString; T = Float64, backend = nothing,
                         rodrigues::Bool = false)
    # The oracle callbacks run on the CPU, so a backend is refused rather
    # than ignored: accepting one would misreport the model as being on the
    # device. `ba_residual` itself is allocation- and branch-free, so a device
    # path is a matter of calling it from a kernel, not of rewriting it.
    backend === nothing || error(
        "ba_oracle_model does not take a backend yet: the port's callbacks " *
        "run on the CPU. ba_residual is GPU-safe, so this is step 4 of the " *
        "roadmap, not a limitation of the port.")
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    obs, rows, isobj = ba_observations(P, T)
    n = length(P.varnames)
    nobs = length(obs)

    c = ExaModels.ExaCore(T)
    c, x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)

    # Hessian sparsity: the lower triangle of each observation's 12x12 block.
    # Observations that share a camera or a point repeat a (row, col) pair, and
    # the two oracle kinds want opposite things from that repetition, so the
    # pattern is built in two forms.
    #
    #   The VECTOR oracle's `hess!` is called once and writes one value per
    #   declared entry -- `y[i] * H[a,b]` for its own observation -- so the
    #   repetition is how an entry accumulates the contributions of every
    #   observation that touches it. Deduplicating it would drop all but one.
    #
    #   The SCALAR oracle has no `hess!`: ExaModels fills its declared pattern
    #   from `hvp!`, one product per column, so every slot carrying the same
    #   (row, col) receives the SAME already-complete second derivative. Those
    #   repeats then sum on assembly and multiply the entry by its multiplicity
    #   -- measured as a 92% error on BA-L1LS's assembled Hessian against the
    #   reference, while `hprod` is exact, because `hprod` never reads the
    #   coordinate path. So the scalar oracle must declare each entry once.
    hr = Vector{Int}(undef, 78 * nobs)
    hc = Vector{Int}(undef, 78 * nobs)
    let p = 0
        for o in obs, a in 1:12, b in 1:a
            p += 1
            i, j = o.idx[a], o.idx[b]
            hr[p], hc[p] = max(i, j), min(i, j)
        end
    end
    if isobj
        seen = Set{Tuple{Int,Int}}()
        keep = [((i, j) in seen ? false : (push!(seen, (i, j)); true))
                for (i, j) in zip(hr, hc)]
        hr, hc = hr[keep], hc[keep]
    end

    if isobj
        # sum of squared residuals
        f = xv -> begin
            s = zero(eltype(xv))
            for o in obs
                s += ba_row(xv, o; rodrigues)^2
            end
            s
        end
        grad! = (g, xv) -> begin
            fill!(g, zero(eltype(g)))
            for o in obs
                r = ba_row(xv, o; rodrigues)
                gr = ba_row_grad(xv, o; rodrigues)
                for a in 1:12
                    g[o.idx[a]] += 2 * r * gr[a]
                end
            end
            g
        end
        # Gauss-Newton plus the residual-curvature term, i.e. the exact Hessian
        # of the sum of squares: 2(J'J + sum_i r_i H_i), contracted with v.
        hvp! = (Hv, xv, v) -> begin
            fill!(Hv, zero(eltype(Hv)))
            for o in obs
                r = ba_row(xv, o; rodrigues)
                gr = ba_row_grad(xv, o; rodrigues)
                H = ba_row_hess(xv, o; rodrigues)
                gv = zero(eltype(Hv))
                for a in 1:12
                    gv += gr[a] * v[o.idx[a]]
                end
                for a in 1:12
                    acc = 2 * gr[a] * gv
                    for b in 1:12
                        acc += 2 * r * H[a, b] * v[o.idx[b]]
                    end
                    Hv[o.idx[a]] += acc
                end
            end
            Hv
        end
        sor = ExaModels.ScalarNonlinearOracle(;
            nvar = n, f = f, grad! = grad!, hvp! = hvp!,
            hess_rows = hr, hess_cols = hc, adapt = Val(true))
        c = ExaModels.objective(c, sor)
    else
        jr = Vector{Int}(undef, 12 * nobs)
        jc = Vector{Int}(undef, 12 * nobs)
        let p = 0
            for (i, o) in enumerate(obs), a in 1:12
                p += 1
                jr[p], jc[p] = i, o.idx[a]
            end
        end
        vor = ExaModels.VectorNonlinearOracle(;
            nvar = n, ncon = nobs,
            jac_rows = jr, jac_cols = jc, hess_rows = hr, hess_cols = hc,
            lcon = zeros(T, nobs), ucon = zeros(T, nobs),
            f! = (cv, xv) -> begin
                for (i, o) in enumerate(obs)
                    cv[i] = ba_row(xv, o; rodrigues)
                end
                cv
            end,
            jac! = (vals, xv) -> begin
                p = 0
                for o in obs
                    gr = ba_row_grad(xv, o; rodrigues)
                    for a in 1:12
                        vals[p += 1] = gr[a]
                    end
                end
                vals
            end,
            hess! = (vals, xv, y) -> begin
                p = 0
                for (i, o) in enumerate(obs)
                    H = ba_row_hess(xv, o; rodrigues)
                    yi = y[i]
                    for a in 1:12, b in 1:a
                        vals[p += 1] = yi * H[a, b]
                    end
                end
                vals
            end,
            adapt = Val(true))
        c = ExaModels.constraint(c, vor)
    end
    return ExaModels.ExaModel(c)
end
