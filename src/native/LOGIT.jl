# AKIVA / HIELOW: a generic Julia port of the HieLoW external Fortran.
#
# WHAT THE FORTRAN IS. Both SIF files declare a single objective group `LIKE`
# holding a single element `ELIKE` with weight -1.0, whose external value is
# `F(B1,B2)` (AKIVA) / `F(B1,B2,B3)` (HIELOW). Both `F`s are thin wrappers
# around ONE routine, `SUBROUTINE HIELOW`, appended to each SIF file together
# with the estimation sample as Fortran `DATA` statements. It is Bierlaire's
# hierarchical (nested) logit log-likelihood:
#
#   * a tree over NNOEUD nodes, `ARBRE(i) = parent(i)`, `-1` at the root, with
#     the guarantee parent > child so one forward sweep suffices;
#   * `THETA(i) = -1` marks an elemental (leaf) node, `-2` the root, and
#     `THETA(i) = k > 0` a structural node whose log-sum scale is the variable
#     `BETA(k)`;
#   * node i's elemental utility is `sum_j BETA(UTCOEF(i,j)) * CARACT(obs,
#     UTCARA(i,j))` over `UTLENG(i)` terms -- linear in the coefficients, with
#     the observation's attributes as data;
#   * every non-leaf node also gets its inclusive value
#     `LOGAR(i) = log( sum_{available children c} exp(UTIL(c)) )`,
#     and a structural node adds `BETA(THETA(i)) * LOGAR(i)` to its own
#     utility;
#   * `DISP(obs,i)` says whether node i was available to observation obs (in
#     HIELOW, 212 of 997 respondents hold no driving licence, so the Car leaf
#     is dropped);
#   * the log-likelihood walks from `CHOSEN(obs)` to the root accumulating
#     `UTIL(node) - LOGAR(parent(node))`, i.e. one conditional-choice log
#     probability per level.
#
# AKIVA is the degenerate case: a 3-node tree, two leaves under the root and
# no structural node, i.e. a plain binary logit over 21 observations in 2
# coefficients. HIELOW is a genuine 2-level nest: leaves 1 (Bus) and 2 (Tram)
# under structural node 3 (Public transport, scale `THE1`), leaf 4 (Car) and
# node 3 under root 5; 997 observations, 3 coefficients.
#
# Because one routine with one data layout covers both, this file ports the
# ALGORITHM and READS THE DATA, rather than transcribing ~7000 numbers: the
# `DATA` statements are parsed out of the SIF file the collection already
# ships. `hielow_loglik` is then a faithful, type-generic translation of the
# value loop for an arbitrary HieLoW tree.
#
# SHAPE: a generic Julia function behind a `ScalarNonlinearOracle`.
# The Fortran depends on the variables, so it cannot be constant-folded the
# way VANDANIUM's B-spline is. And the element's VALUE is a sum over all
# observations -- 21 terms for AKIVA, 997 for HIELOW -- so as an ExaModels
# expression it would be hundreds of operations deep, far past the ~30-deep
# chain that did not finish compiling in an hour on BA. Derivatives come from
# ForwardDiff over `hielow_loglik`; the Fortran's hand-coded `GR`/`HESS`
# recursions (Daly's equations 6/21/23 and the Hessian sweep) are NOT ported.
#
# OVERFLOW, AND WHERE THIS PORT DIFFERS FROM THE FORTRAN. The Fortran forms
# `exp(UTIL)` and sums it, so it guards by SATURATION: utilities are clipped
# at `LOGGRA = log(1.0D308) = 709.1962086421661`, `exp` of anything at or
# above that is replaced by `GRAND = 1.0D308`, and the running sum is clamped
# to `GRAND`. This port instead computes the same inclusive value by shifting
# by the maximum,
#
#     LOGAR = m + log1p( sum_{c != argmax} exp(UTIL(c) - m) ),   m = max UTIL(c)
#
# which cannot overflow and needs no clipping, and which is exact (`log1p` of
# 0, no rounding at all) for the single-available-child case that 212 of
# HIELOW's observations hit. In the normal regime the two agree to roundoff.
# They DIFFER once any utility reaches 709.1962: the Fortran's log-likelihood
# saturates -- each level of the choice path can contribute no less than
# about `-LOGGRA` -- while this port keeps returning the true (unbounded)
# value. Worth stating plainly: in that regime the Fortran is internally
# inconsistent, because its `DERIV`/`HESS` recursions are the derivatives of
# the UNCLIPPED model and know nothing about the clamp. So this port's value
# is the one its own derivatives belong to, and the Fortran's is not.
# `|BETA| <= ~7` keeps HIELOW's utilities inside the safe range (its largest
# attribute is 100), so the difference is unreachable from any sane iterate.
#
# A PRECISION TRAP, and the reason HIELOW first missed by 1e-10. AKIVA writes
# every attribute with an explicit double exponent (`52.9D0`); HIELOW writes
# all 5982 of them bare (`3.8712`, `0.693147`). A bare real constant in Fortran
# is DEFAULT REAL, i.e. SINGLE precision, so the `DOUBLE PRECISION CARACT`
# array is loaded with `Float64(Float32(3.8712)) = 3.8712000846862793`: the
# sample the Fortran actually fits is only single-precision accurate. Parsing
# those literals as Float64 reproduces HIELOW's objective to 8e-11 relative
# and its gradient to 1.4e-9; rounding them through Float32, as Fortran does,
# takes both to 1e-15. `_hielow_float` therefore rounds exactly the literals
# Fortran would round -- the same class of bug as VANDANIUM's undeclared `xx`.
# Integer literals (`26`) convert exactly and are left alone.

"""
    HieLoWData

The estimation sample and tree of one HieLoW problem, as the Fortran's `DATA`
statements declare it. Field names are the Fortran's: `arbre[i]` is node `i`'s
parent (`-1` at the root), `theta[i]` is `-1` for a leaf, `-2` for the root and
`k > 0` for a structural node scaled by `beta[k]`, `utcoef`/`utcara`/`utleng`
describe each node's elemental utility, `chosen[obs]` is the observed choice,
`caract[obs, :]` the attributes and `disp[obs, i]` the availability mask.
`children[i]` is derived from `arbre` for the inclusive-value sweep.
"""
struct HieLoWData
    nnoeud::Int
    ncoef::Int
    nobs::Int
    arbre::Vector{Int}
    theta::Vector{Int}
    utleng::Vector{Int}
    utcoef::Matrix{Int}
    utcara::Matrix{Int}
    chosen::Vector{Int}
    caract::Matrix{Float64}
    disp::Matrix{Bool}
    children::Vector{Vector{Int}}
end

# `DATA NAME(i) /v/` or `DATA NAME(i,j) /v/`, with or without space before `/`.
const _HIELOW_DATA_RE =
    r"^\s*DATA\s+([A-Z]+)\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/\s*([^/]*?)\s*/"
# `DISP(i,j) = .FALSE.`
const _HIELOW_DISP_RE =
    r"^\s*DISP\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*\.(TRUE|FALSE)\."

"""
    _hielow_float(s) -> Float64

The value a Fortran `DATA` statement gives a `DOUBLE PRECISION` array for the
literal `s`, INCLUDING the precision loss Fortran's default typing imposes.

Three cases, and the third is the trap (see this file's header):
  * `52.9D0` -- a `D` exponent makes it a double-precision constant;
  * `26` -- an integer constant, converted to double exactly;
  * `3.8712` -- NO exponent letter, so it is a DEFAULT REAL (single precision)
    constant, rounded to `Float32` before being widened to double. AKIVA
    writes every attribute with `D0`; HIELOW writes all 5982 of them this way,
    so its sample is only single-precision accurate.
"""
function _hielow_float(s::AbstractString)
    if occursin('D', s) || occursin('d', s)
        return parse(Float64, replace(s, 'D' => 'e', 'd' => 'e'))
    end
    v = parse(Float64, s)
    # An integer constant converts to double exactly. A real constant with no
    # exponent letter is DEFAULT REAL -- single precision -- so round it.
    return any(c -> c in ('.', 'e', 'E'), s) ? Float64(Float32(v)) : v
end

"""
    hielow_parse(path) -> HieLoWData

Read the HieLoW tree and estimation sample out of the Fortran appended to a
SIF file. Only `PARAMETER`, `DATA` and the `DISP(i,j) = .FALSE.` overrides are
read -- never the function body, which is ported as code in
[`hielow_loglik`](@ref).

Everything it needs is stated twice in the file (sizes in `PARAMETER`, entries
in `DATA`), so the two are cross-checked and a shortfall is an error rather
than a silently truncated sample.
"""
function hielow_parse(path::AbstractString)
    txt = read(path, String)
    par(name) = begin
        m = match(Regex("\\b" * name * "\\s*=\\s*(\\d+)"), txt)
        m === nothing && error("$path: no PARAMETER $name in the appended Fortran")
        parse(Int, m.captures[1])
    end
    nnoeud, nmaxut, nobs, ncarac, ncoef = par.(("NNOEUD", "NMAXUT", "NOBS", "NCARAC", "NCOEF"))

    arbre = zeros(Int, nnoeud)
    theta = zeros(Int, nnoeud)
    utleng = fill(-1, nnoeud)
    utcoef = zeros(Int, nnoeud, nmaxut)
    utcara = zeros(Int, nnoeud, nmaxut)
    chosen = zeros(Int, nobs)
    caract = fill(NaN, nobs, ncarac)
    disp = trues(nobs, nnoeud)          # `DATA DISP/NTOT*.TRUE./`
    seen = Dict("ARBRE" => 0, "THETA" => 0, "UTLENG" => 0, "CHOSEN" => 0, "CARACT" => 0)

    for ln in eachline(IOBuffer(txt))
        startswith(ln, "*") && continue
        m = match(_HIELOW_DATA_RE, ln)
        if m !== nothing
            name, i = m.captures[1], parse(Int, m.captures[2])
            j = m.captures[3] === nothing ? 0 : parse(Int, m.captures[3])
            v = m.captures[4]
            if name == "ARBRE"
                arbre[i] = parse(Int, v); seen[name] += 1
            elseif name == "THETA"
                theta[i] = parse(Int, v); seen[name] += 1
            elseif name == "UTLENG"
                utleng[i] = parse(Int, v); seen[name] += 1
            elseif name == "UTCOEF"
                utcoef[i, j] = parse(Int, v)
            elseif name == "UTCARA"
                utcara[i, j] = parse(Int, v)
            elseif name == "CHOSEN"
                chosen[i] = parse(Int, v); seen[name] += 1
            elseif name == "CARACT"
                caract[i, j] = _hielow_float(v); seen[name] += 1
            end
            continue
        end
        m = match(_HIELOW_DISP_RE, ln)
        m === nothing && continue
        disp[parse(Int, m.captures[1]), parse(Int, m.captures[2])] =
            m.captures[3] == "TRUE"
    end

    seen["ARBRE"] == nnoeud || error("$path: $(seen["ARBRE"]) of $nnoeud ARBRE entries")
    seen["THETA"] == nnoeud || error("$path: $(seen["THETA"]) of $nnoeud THETA entries")
    seen["UTLENG"] == nnoeud || error("$path: $(seen["UTLENG"]) of $nnoeud UTLENG entries")
    seen["CHOSEN"] == nobs || error("$path: $(seen["CHOSEN"]) of $nobs CHOSEN entries")
    seen["CARACT"] == nobs * ncarac ||
        error("$path: $(seen["CARACT"]) of $(nobs * ncarac) CARACT entries")
    any(isnan, caract) && error("$path: CARACT has holes after parsing")

    children = [Int[] for _ in 1:nnoeud]
    nroot = 0
    for i in 1:nnoeud
        if arbre[i] == -1
            nroot += 1
        else
            arbre[i] > i || error("$path: ARBRE($i) = $(arbre[i]) breaks parent > child")
            push!(children[arbre[i]], i)
        end
    end
    nroot == 1 || error("$path: $nroot roots in ARBRE")

    # Shapes the port relies on, checked rather than assumed. A non-leaf with
    # no available child would make its inclusive value log(0); the Fortran
    # would propagate -Inf, and rather than reproduce that quietly we refuse.
    for obs in 1:nobs, i in 1:nnoeud
        theta[i] == -1 && continue
        any(c -> disp[obs, c], children[i]) ||
            error("$path: node $i has no available child at observation $obs")
    end
    for obs in 1:nobs
        disp[obs, chosen[obs]] ||
            error("$path: observation $obs chose unavailable node $(chosen[obs])")
    end
    for i in 1:nnoeud
        0 <= utleng[i] <= nmaxut || error("$path: UTLENG($i) = $(utleng[i])")
        theta[i] <= ncoef || error("$path: THETA($i) = $(theta[i]) exceeds NCOEF")
    end

    return HieLoWData(nnoeud, ncoef, nobs, arbre, theta, utleng, utcoef, utcara,
                      chosen, caract, disp, children)
end

const _HIELOW_DATA_CACHE = Dict{String,HieLoWData}()

"Parsed sample for a problem name, cached: HIELOW's Fortran is ~10800 lines."
hielow_data(name::AbstractString) =
    get!(() -> hielow_parse(sif_problem_path(name)), _HIELOW_DATA_CACHE, String(name))

"""
    hielow_loglik(beta, d::HieLoWData) -> T

Log-likelihood of the hierarchical logit model `d` at coefficients `beta`: a
translation of `SUBROUTINE HIELOW`'s value loop, for an arbitrary tree.

Generic in `eltype(beta)`, so the same code evaluates in `Float64`, `Float32`
or `BigFloat`, and in `ForwardDiff.Dual` for the derivatives. Inclusive values
are formed by shifting by the maximum (see this file's header), so nothing
overflows and the Fortran's `LOGGRA`/`GRAND` clipping is not needed.

The two length-`NNOEUD` scratch vectors (3 and 5 elements for the two shipped
problems) are the only allocation, and they are allocated once per call, not
once per observation.
"""
function hielow_loglik(beta::AbstractVector{T}, d::HieLoWData) where {T}
    length(beta) == d.ncoef ||
        throw(DimensionMismatch("expected $(d.ncoef) coefficients, got $(length(beta))"))
    return hielow_loglik!(Vector{T}(undef, d.nnoeud), Vector{T}(undef, d.nnoeud),
                          beta, d)
end

"In-place core of [`hielow_loglik`](@ref), with caller-supplied scratch."
function hielow_loglik!(util::AbstractVector{T}, logar::AbstractVector{T},
                        beta::AbstractVector, d::HieLoWData) where {T}
    like = zero(T)
    @inbounds for obs in 1:d.nobs
        for nd in 1:d.nnoeud
            # Elemental utility: linear in the coefficients. The root has none
            # (the Fortran skips the block when ARBRE(NOEUD) == -1), and a
            # structural node has UTLENG == 0, so both start from zero.
            u = zero(T)
            if d.arbre[nd] != -1
                for i in 1:d.utleng[nd]
                    u += T(beta[d.utcoef[nd, i]]) * T(d.caract[obs, d.utcara[nd, i]])
                end
            end
            if d.theta[nd] != -1
                # Inclusive value over the available children, max-shifted.
                kmax = 0
                m = zero(T)
                for c in d.children[nd]
                    d.disp[obs, c] || continue
                    if kmax == 0 || util[c] > m
                        kmax, m = c, util[c]
                    end
                end
                s = zero(T)
                for c in d.children[nd]
                    (c == kmax || !d.disp[obs, c]) && continue
                    s += exp(util[c] - m)
                end
                logar[nd] = m + log1p(s)
                if d.theta[nd] > 0
                    u += T(beta[d.theta[nd]]) * logar[nd]
                end
            end
            util[nd] = u
        end
        # One conditional log-probability per level, from the chosen leaf up.
        nd = d.chosen[obs]
        while d.arbre[nd] != -1
            like += util[nd] - logar[d.arbre[nd]]
            nd = d.arbre[nd]
        end
    end
    return like
end

# ---------------------------------------------------------------------------
# Interfacing the generic function to ExaModels as an external oracle.
#
# Pure assembly below: the mathematics is all in hielow_loglik, and every
# derivative is ForwardDiff over it. Both problems are unconstrained with a
# single objective group of trivial type, weight w on the one element, so the
# objective is w * loglik - constant; the group fields are checked rather than
# assumed, because a silent shape change would be a silently wrong model.

"""
    logit_oracle_model(name; T = Float64, backend = nothing) -> ExaModel

Build AKIVA or HIELOW as an `ExaModel` whose objective callbacks are
[`hielow_loglik`](@ref) rather than the compiled HieLoW Fortran.

Needs no `sifdecoder` and no Fortran toolchain: the variables and bounds come
from `sif_decode(...; external = :defer)` and the sample from
[`hielow_parse`](@ref).

!!! warning "Second derivatives need the with-`y` NLPModels methods"
    ExaModels 0.12.1 adds a `ScalarNonlinearOracle`'s Hessian only in
    `hprod!(m, x, y, v, Hv)` and `hess_coord!(m, x, y, h)`. The no-`y`
    methods, which is what `NLPModels.hess(nlp, x)` and
    `NLPModels.hprod(nlp, x, v)` call for an UNCONSTRAINED model, fall
    through to the symbolic-only path and return ZERO here. Pass
    `y = zeros(0)` explicitly. This is upstream and pre-existing -- the
    `BA-L*LS` native models behave the same way -- not a gap in this port;
    `test/../logit_verify.jl` measures both paths.

    Also upstream, and also shared with `BA-L*`: `T = Float32` does not build,
    because ExaModels' oracle work buffers are typed `Vector{Float64}`.
    [`hielow_loglik`](@ref) itself is fully generic -- `Float32` and
    `BigFloat` evaluations are part of the verification -- so this is the
    plumbing, not the port.
"""
function logit_oracle_model(name::AbstractString; T = Float64, backend = nothing)
    # The oracle's callbacks are CPU Julia. hielow_loglik is type-generic, so
    # a device path is roadmap step 4 rather than a rewrite -- but accepting a
    # backend here would misreport where the model runs.
    backend === nothing || error(
        "logit_oracle_model does not take a backend yet: the port's callbacks " *
        "run on the CPU. Accepting one would misreport the model as being on " *
        "the device.")
    P = ExaModelsSIF.sif_decode(sif_problem_path(name); external = :defer)
    haskey(P.external_types, "LIKE") ||
        error("$name: expected a deferred LIKE element type, found " *
              "$(collect(keys(P.external_types)))")
    length(P.grouporder) == 1 ||
        error("$name: expected 1 group, found $(length(P.grouporder))")
    g = P.groups[only(P.grouporder)]
    (g.kind == 'N' && isempty(g.lin) && length(g.elems) == 1 && g.scale == 1 &&
     g.range_ === nothing) || error("$name: unexpected objective group shape")
    en, w = only(g.elems)
    e = P.elements[en]
    d = hielow_data(name)

    # BETA(i) of the Fortran is the element's i-th elemental variable B<i>.
    idx = ntuple(i -> e.varmap[Symbol(:B, i)], d.ncoef)
    length(unique(idx)) == d.ncoef ||
        error("$name: B1..B$(d.ncoef) do not map to distinct variables")

    n = length(P.varnames)
    wT, cT = T(w), T(g.constant)
    gather(x) = [x[i] for i in idx]
    f = x -> wT * hielow_loglik(gather(x), d) - cT

    c = ExaModels.ExaCore(T)
    c, x = ExaModels.add_var(c, n; start = P.x0, lvar = P.lvar, uvar = P.uvar)

    # ncoef is 2 or 3, so the objective block is dense and tiny.
    hr = [idx[a] >= idx[b] ? idx[a] : idx[b] for a in 1:d.ncoef for b in 1:a]
    hc = [idx[a] >= idx[b] ? idx[b] : idx[a] for a in 1:d.ncoef for b in 1:a]

    grad! = (gv, xv) -> begin
        fill!(gv, zero(eltype(gv)))
        gb = ForwardDiff.gradient(b -> hielow_loglik(b, d), gather(xv))
        for a in 1:d.ncoef
            gv[idx[a]] += wT * gb[a]
        end
        gv
    end
    hvp! = (Hv, xv, v) -> begin
        fill!(Hv, zero(eltype(Hv)))
        H = ForwardDiff.hessian(b -> hielow_loglik(b, d), gather(xv))
        for a in 1:d.ncoef, b in 1:d.ncoef
            Hv[idx[a]] += wT * H[a, b] * v[idx[b]]
        end
        Hv
    end
    c = ExaModels.objective(c, ExaModels.ScalarNonlinearOracle(;
        nvar = n, f = f, grad! = grad!, hvp! = hvp!,
        hess_rows = hr, hess_cols = hc, adapt = Val(true)))
    return ExaModels.ExaModel(c)
end

"""
    akiva_model(; T = Float64, backend = nothing) -> ExaModel

AKIVA -- Ben-Akiva and Lerman's 21-observation binary logit in 2 coefficients
-- as a native `ExaModel`. See [`logit_oracle_model`](@ref).
"""
akiva_model(; kwargs...) = logit_oracle_model("AKIVA"; kwargs...)

"""
    hielow_model(; T = Float64, backend = nothing) -> ExaModel

HIELOW -- HieLoW's 997-observation two-level nested logit in 3 coefficients
(two taste coefficients and the nest scale `THE1`) -- as a native `ExaModel`.
See [`logit_oracle_model`](@ref).
"""
hielow_model(; kwargs...) = logit_oracle_model("HIELOW"; kwargs...)
