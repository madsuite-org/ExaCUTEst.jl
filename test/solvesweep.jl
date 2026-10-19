# test/solvesweep.jl — can every problem actually be SOLVED?
#
# Decoding and validating a model says it is correct; it does not say a solver
# can get through it. This sweep closes that gap: build each problem and run
# MadNLP with a wall-time cap, recording the status.
#
# Sharded and restartable like valsweep.jl. A problem whose single evaluation
# outlasts the driver's process timeout is attributed from current.<shard>.txt
# on the next pass, so one wedged Hessian costs one problem, not the sweep.
#
# Needs MadNLP, which is not a dependency of this package: run it in an
# environment that has both, e.g. one that `Pkg.develop`s this checkout and
# adds MadNLP.
using ExaCUTEst, MadNLP, NLPModels
using ExaCUTEst: ExaCUTEstModel, list_sif_problems, classification

const SHARD = parse(Int, get(ENV, "SWEEP_SHARD", "0"))
const NSHARDS = parse(Int, get(ENV, "SWEEP_NSHARDS", "1"))
const CAP = parse(Float64, get(ENV, "SOLVE_CAP", "30.0"))
const RUNS = get(ENV, "SWEEP_RUNS", joinpath(@__DIR__, "..", "runs"))
const OUT = joinpath(RUNS, "solve_results.s$SHARD.tsv")
const CUR = joinpath(RUNS, "solve_current.s$SHARD.txt")
const CHUNK = 40

mkpath(RUNS)
done = Set{String}()
isfile(OUT) && for ln in eachline(OUT)
    push!(done, String(first(split(ln, '\t'))))
end
# a problem in flight when the process died: attribute it, do not retry
if isfile(CUR)
    prev = strip(read(CUR, String))
    if !isempty(prev) && !(prev in done)
        open(OUT, "a") do io
            println(io, prev, "\tKILLED\t\t\t(process died: a single build or " *
                        "evaluation outlasted the driver timeout)")
        end
        push!(done, String(prev))
    end
end

function run_sweep()
    d = classification()
    names = list_sif_problems()
    # smallest first, so a shard yields information early
    sort!(names; by = p -> haskey(d, p) ?
          d[p]["variables"]["number"] + d[p]["constraints"]["number"] : typemax(Int))
    nproc = 0
    open(OUT, "a") do io
        for (i, name) in enumerate(names)
            (i - 1) % NSHARDS == SHARD || continue
            name in done && continue
            write(CUR, name)
            status, obj_, iter, tb, ts = "", "", "", "", ""
            try
                t = @elapsed m = ExaCUTEstModel(name)
                tb = string(round(t, sigdigits = 3))
                t2 = @elapsed res = madnlp(m; max_wall_time = CAP, max_iter = 3000,
                                           print_level = MadNLP.ERROR)
                ts = string(round(t2, sigdigits = 3))
                status = string(res.status)
                obj_ = string(round(res.objective, sigdigits = 10))
                iter = string(res.iter)
            catch err
                status = "ERROR"
                obj_ = replace(first(sprint(showerror, err), 110), '\t' => ' ', '\n' => ' ')
            end
            println(io, join([name, status, obj_, iter, tb, ts], '\t'))
            flush(io)
            rm(CUR; force = true)
            nproc += 1
            # exit periodically so the driver reclaims memory the solver and
            # the model builders have accumulated
            nproc >= CHUNK && exit(42)
        end
    end
    return nothing
end

run_sweep()
println("shard $SHARD complete")
