# Does the VANDANIUM* port reproduce the Fortran, and is the undeclared `xx`
# single precision as Fortran's implicit typing says? Decided by measurement.
#
# The element is A * b(...) with b a constant, so each row is LINEAR: row i =
# sum_k BK_ik * A_ik - const. Compare against CUTEst's cons/obj at x0.
using Pkg; Pkg.activate("/home/sushin/git/pkg/ExaCUTEst.jl")
using ExaCUTEst, ExaModelsSIF, NLPModels, LinearAlgebra
using FileWatching.Pidfile: mkpidlock
import CUTEst
include("/home/sushin/git/pkg/ExaCUTEst.jl/src/native/VANDANIUM.jl")
const B = "/home/sushin/.julia/artifacts/1fa01d4c3b344b8cd32b8ef78c0abeac9bb9cce2"

for name in ("VANDANIUMS", "VANDANMSLS")
    P = ExaModelsSIF.sif_decode(joinpath(B, name * ".SIF"); external = :defer)
    lk = mkpidlock("/tmp/cutest-decode.pid"; wait = true)
    ref = try CUTEst.CUTEstModel(name) finally close(lk) end
    n = ref.meta.nvar
    status = zeros(Cint, 1); vname = Matrix{Cchar}(undef, 10, n)
    CUTEst.varnames(Float64, ref.libsif, status, Cint[n], vname)
    rn = [String(strip(join(Char(UInt8(c)) for c in vname[:, i]))) for i in 1:n]
    x = copy(ref.meta.x0)
    ourpos = Dict(P.varnames[i] => i for i in eachindex(P.varnames))
    xo = zeros(length(P.varnames)); for (k, s) in enumerate(rn); xo[ourpos[s]] = x[k]; end
    rows = [g for g in P.grouporder if P.groups[g].kind != 'N']
    isobj = isempty(rows); isobj && (rows = copy(P.grouporder))

    for single in (true, false)
        vals = Dict{String,Float64}()
        for g in rows
            grp = P.groups[g]
            s = -grp.constant
            for (en, w) in grp.elems
                e = P.elements[en]
                pv = e.parvals
                bk = vandanium_bspline(pv[:X0], pv[:K], pv[:H], pv[:X]; single_xx = single)
                s += w * bk * xo[e.varmap[:A]]
            end
            for (vi, a) in grp.lin
                s += a * xo[vi]
            end
            vals[g] = s
        end
        detail = ""
        if ref.meta.ncon > 0
            mc = ref.meta.ncon
            pname = Vector{Cchar}(undef, 10); cname = Matrix{Cchar}(undef, 10, mc)
            CUTEst.cnames(Float64, ref.libsif, status, Cint[n], Cint[mc], pname, vname, cname)
            cr = [String(strip(join(Char(UInt8(c)) for c in cname[:, i]))) for i in 1:mc]
            ca = NLPModels.cons(ref, x); ours = [vals[s] for s in cr]
            e = maximum(abs.(ca .- ours) ./ (1 .+ max.(abs.(ca), abs.(ours))))
            detail = "cons relerr $(round(e, sigdigits = 3))"
        else
            fa = NLPModels.obj(ref, x)
            gt = P.gtypes
            fb = sum(v -> v^2, values(vals))   # SUR2: sum of squares
            e = abs(fa - fb) / (1 + max(abs(fa), abs(fb)))
            detail = "obj ref=$fa ours=$fb relerr=$(round(e, sigdigits = 3))"
        end
        println(rpad(name, 12), " single_xx=", rpad(single, 6), " ", detail)
    end
    finalize(ref)
end
