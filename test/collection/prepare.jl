# Prepare a run directory: write the population and the provenance of every
# package under test, then check that a model builds and evaluates on the device.
#
#   CV_RUN=<run dir> julia --project=test/gpu test/collection/prepare.jl
#
# population.tsv lists every problem of the collection with its tier (native
# or decoded); provenance.txt records where each package loads from and, for a
# git checkout, its commit and whether the tree is dirty. The run measures
# whatever these paths hold while it runs: keep them unchanged until it ends.
include(joinpath(@__DIR__, "..", "gpu", "activate.jl"))
using ExaCUTEst, ExaModels, ExaModelsSIF, CUTEst, NLPModels, CUDA
using Pkg

const F = ENV["CV_RUN"]
for d in ("results", "state", "logs", "shards")
    mkpath(joinpath(F, d))
end

function gitrev(dir)
    try
        git(args...) = readchomp(pipeline(`git -C $dir $args`; stderr = devnull))
        rev = git("rev-parse", "HEAD")
        dirty = !isempty(git("status", "--porcelain", "--untracked-files=no"))
        return rev * (dirty ? " (dirty)" : "")
    catch
        return "-"
    end
end

# A package resolved from a registry or a `rev` source is identified by its
# Manifest tree hash (and revision), not by a git checkout.
function manifest_id(m)
    info = get(Pkg.dependencies(), Base.PkgId(m).uuid, nothing)
    info === nothing && return "-"
    rev = info.git_revision === nothing ? "" : " rev=$(info.git_revision)"
    return "tree=$(something(info.tree_hash, "-"))$rev"
end

open(joinpath(F, "provenance.txt"), "w") do io
    for m in (ExaCUTEst, ExaModels, ExaModelsSIF, CUTEst, CUDA)
        d = pkgdir(m)
        println(io, rpad(string(m), 14), d, "\t", gitrev(d), "\t", manifest_id(m),
                "\tv", pkgversion(m))
    end
    println(io, "CUTEst.libsif_path  ", CUTEst.libsif_path)
    println(io, "MASTSIF             ", ExaCUTEst.mastsif())
    println(io, "julia               ", VERSION)
    println(io, "prepared            ", gethostname(), " ", round(Int, time()))
end
print(read(joinpath(F, "provenance.txt"), String))

probs = sort(ExaCUTEst.list_sif_problems())
nat = Set(ExaCUTEst.native_problems())
issubset(nat, probs) || error("native problems outside the population: $(setdiff(nat, probs))")
open(joinpath(F, "population.tsv"), "w") do io
    println(io, "problem\ttier")
    for p in probs
        println(io, p, '\t', p in nat ? "native" : "decoded")
    end
end
println("population: $(length(probs)) problems, $(length(nat)) native")

CUDA.functional() || error("CUDA not functional on $(gethostname())")
CUDA.allowscalar(false)
nlp = ExaCUTEstModel("ROSENBR"; backend = CUDABackend())
println("ROSENBR on device: obj(x0) = ", obj(nlp, nlp.meta.x0))
