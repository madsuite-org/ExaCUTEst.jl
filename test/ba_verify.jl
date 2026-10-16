# test/ba_verify.jl -- roadmap step 3 for the BA-L* family: does the generic
# Julia port, interfaced as an ExaModels oracle, reproduce the Fortran?
#
#   julia --project=. test/ba_verify.jl
#
# This verification is INDEPENDENT, unlike test/oracle_assembly.jl: two
# genuinely different implementations rather than two callers of the same
# compiled routine. That independence is what caught the legacy HS67 port.
#
# Does the Julia-port oracle model match the Fortran on every callback?
# obj, grad, cons, jac and hprod, at x0 and a perturbed point, name-permuted.
using ExaCUTEst, ExaModelsSIF, ExaModels, NLPModels, ForwardDiff
using SparseArrays, LinearAlgebra, Random
using FileWatching.Pidfile: mkpidlock
import CUTEst
log = open(get(ENV, "OUT_TSV",
    joinpath(@__DIR__, "..", "runs", "ba_verify.tsv")), "w")
println(log, "problem\tverdict\tdetail"); flush(log)
rel(a, b) = begin
    m = isfinite.(a) .& isfinite.(b)
    any(m) ? maximum(abs.(a[m] .- b[m]) ./ (1 .+ max.(abs.(a[m]), abs.(b[m])))) : NaN
end
for name in ("BA-L1", "BA-L1LS", "BA-L49", "BA-L49LS", "BA-L73", "BA-L73LS",
             "BA-L21", "BA-L21LS", "BA-L16", "BA-L16LS", "BA-L52", "BA-L52LS")
    try
        t = @elapsed m = ExaCUTEst.ba_oracle_model(name)
        lk = mkpidlock("/tmp/cutest-decode.pid"; wait = true)
        ref = try CUTEst.CUTEstModel(name) finally close(lk) end
        n, mc = ref.meta.nvar, ref.meta.ncon
        status = zeros(Cint, 1); vname = Matrix{Cchar}(undef, 10, n)
        CUTEst.varnames(Float64, ref.libsif, status, Cint[n], vname)
        rn = [String(strip(join(Char(UInt8(c)) for c in vname[:, i]))) for i in 1:n]
        P = ExaModelsSIF.sif_decode(ExaCUTEst.sif_problem_path(name); external = :defer)
        vp = [P.varidx[s] for s in rn]
        perm = Int[]
        if mc > 0
            pname = Vector{Cchar}(undef, 10); cname = Matrix{Cchar}(undef, 10, mc)
            CUTEst.cnames(Float64, ref.libsif, status, Cint[n], Cint[mc], pname, vname, cname)
            cr = [String(strip(join(Char(UInt8(c)) for c in cname[:, i]))) for i in 1:mc]
            _, rows, _ = ExaCUTEst.ba_observations(P)
            rowof = Dict(g => i for (i, g) in enumerate(rows))
            perm = [rowof[s] for s in cr]
        end
        msgs = String[]
        Random.seed!(7)
        for (tag, xr) in (("x0", copy(ref.meta.x0)), ("pert", ref.meta.x0 .+ 0.01 .* randn(n)))
            xm = zeros(length(P.varnames)); xm[vp] = xr
            fa, fb = NLPModels.obj(ref, xr), obj(m, xm)
            e = abs(fa - fb) / (1 + max(abs(fa), abs(fb)))
            e <= 1e-10 || push!(msgs, "obj@$tag ref=$fa ours=$fb")
            e = rel(NLPModels.grad(ref, xr), grad(m, xm)[vp])
            e <= 1e-9 || push!(msgs, "grad@$tag $(round(e, sigdigits=3))")
            if mc > 0
                e = rel(NLPModels.cons(ref, xr), cons(m, xm)[perm])
                e <= 1e-10 || push!(msgs, "cons@$tag $(round(e, sigdigits=3))")
                Ja = sparse(NLPModels.jac_structure(ref)..., NLPModels.jac_coord(ref, xr), mc, n)
                Jb = sparse(jac_structure(m)..., jac_coord(m, xm), mc, length(P.varnames))[perm, vp]
                e = maximum(abs, Ja - Jb; init = 0.0) / (1 + maximum(abs, Ja; init = 0.0))
                e <= 1e-9 || push!(msgs, "jac@$tag $(round(e, sigdigits=3))")
            end
            y = mc > 0 ? randn(mc) : Float64[]
            ym = zeros(mc); mc > 0 && (ym[perm] = y)
            v = randn(n); vm = zeros(length(P.varnames)); vm[vp] = v
            e = rel(NLPModels.hprod(ref, xr, y, v), hprod(m, xm, ym, vm)[vp])
            e <= 1e-8 || push!(msgs, "hprod@$tag $(round(e, sigdigits=3))")
        end
        println(log, join([name, isempty(msgs) ? "ORACLE_MATCHES" : "DIFFERS",
                 isempty(msgs) ? "built in $(round(t, digits=1)) s; all callbacks match" :
                 join(msgs, " | ")], '\t'))
        flush(log); finalize(ref)
    catch err
        println(log, join([name, "ERROR",
                 replace(first(sprint(showerror, err), 130), '\t'=>' ', '\n'=>' ')], '\t'))
        flush(log)
    end
end
close(log)
