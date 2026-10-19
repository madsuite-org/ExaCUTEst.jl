# One long-lived verification worker. Started by run_worker.sh, which sets:
#
#   CV_RUN     run directory (results/, state/, shards/, population.tsv)
#   CV_MODE    gpu | cpu
#   CV_WORKER  worker name, unique across hosts (gpu0, cpu-<host>-07, ...)
#   CV_SHARD   gpu only: problem list, relative to CV_RUN
#
# and optionally CV_TIMEOUT (s per problem, 3600), CV_TIMEOUT_REF (s per
# CUTEst build, 3600), CV_ASM_MAX (largest pattern assembled to a sparse
# matrix, 1e8), CV_TSV (configuration rows file in results/), CV_CUTEST_LOCK.
#
# gpu: for each problem of its shard, GPU Float64 against CPU Float64 and GPU
#      Float32 against CPU Float32 (results/gpu.tsv). One process per device, so
#      kernels compiled for one problem serve every later problem sharing their
#      type signatures; shards are grouped by signature for that reason.
# cpu: claims problems from one queue shared by every CPU worker on every host;
#      for each, CPU Float32 against CPU Float64 (results/cpu.tsv), then CPU
#      Float64 against the compiled Fortran (results/fortran.tsv).
#
# Every row is appended under flock the moment it exists. The problem and phase
# in flight are written to state/inflight.<worker> before each phase, and
# alarm(2) bounds each problem: SIGALRM kills the process (exit 142) and the
# supervising run_worker.sh records the culprit and restarts the worker, which
# resumes after it. CUTEstModel construction, on every host, holds an exclusive
# flock on CV_CUTEST_LOCK (default CV_RUN/cutest.lock): CUTEst decodes and
# compiles every problem in one shared build directory, and concurrent builds
# there corrupt each other.

include(joinpath(@__DIR__, "..", "gpu", "activate.jl"))

const MODE = ENV["CV_MODE"]
if MODE == "gpu"
    using CUDA
    CUDA.allowscalar(false)        # before anything else; never relaxed
end
using Printf, Random, LinearAlgebra, SparseArrays
using ExaCUTEst, ExaModels, NLPModels, CUTEst

const FR = ENV["CV_RUN"]
const WID = ENV["CV_WORKER"]
const HOST = gethostname()
const TMO = parse(Int, get(ENV, "CV_TIMEOUT", "3600"))          # per problem
const TMO_REF = parse(Int, get(ENV, "CV_TIMEOUT_REF", "3600"))  # one CUTEst build
const ASM_MAX = parse(Int, get(ENV, "CV_ASM_MAX", "100000000"))
const TSV_CFG = joinpath(FR, "results", get(ENV, "CV_TSV", MODE == "gpu" ? "gpu.tsv" : "cpu.tsv"))
const TSV_A = joinpath(FR, "results", "fortran.tsv")
const CUTEST_LOCK = get(ENV, "CV_CUTEST_LOCK", joinpath(FR, "cutest.lock"))
const NATIVE = Set(ExaCUTEst.native_problems())
const CUR = Ref("")
tier() = CUR[] in NATIVE ? "native" : "decoded"

# --- locking, rows, watchdog --------------------------------------------------

const LOCK_EX = Cint(2)
"Run `f` holding an exclusive flock on `path` (released by the kernel if this process dies)."
function withflock(f, path)
    io = open(path, "a")
    try
        r = @ccall gc_safe = true flock(reinterpret(Cint, fd(io))::Cint, LOCK_EX::Cint)::Cint
        r == 0 || error("flock($path) failed, errno $(Libc.errno())")
        return f()
    finally
        close(io)
    end
end

append_row(path, row) = withflock(path * ".lock") do
    open(io -> println(io, row), path, "a")
end

arm(t) = ccall(:alarm, Cuint, (Cuint,), t)

function with_cutest_lock(f)
    arm(0)                          # waiting for the lock is not the problem's time
    withflock(CUTEST_LOCK) do
        arm(TMO_REF)
        r = f()
        arm(TMO)
        r
    end
end

inflight(phase) = write(joinpath(FR, "state", "inflight.$WID"),
                        "$(CUR[])\t$phase\t$(round(Int, time()))\n")

clean(s) = strip(replace(string(s), '\t' => ' ', '\n' => ' ', '\r' => ' '))

function emit(; cmp, callback, point = "-", status, n = -1, scale = NaN,
              absdiff = NaN, ulp = NaN, nfin_ref = -1, nfin_test = -1,
              nfin_same = "-", nnz_ref = -1, nnz_test = -1, secs = NaN, note = "")
    row = join((CUR[], tier(), MODE, cmp, callback, point, status, n, scale,
                absdiff, ulp, nfin_ref, nfin_test, nfin_same, nnz_ref, nnz_test,
                @sprintf("%.2f", secs), HOST, WID, clean(note)), '\t')
    append_row(TSV_CFG, row)
end

include(joinpath(@__DIR__, "compare.jl"))
MODE == "cpu" && include(joinpath(@__DIR__, "fortran.jl"))

# --- one problem ------------------------------------------------------------

function run_gpu(prob)
    nlp64 = build(Float64, nothing, "cpu_f64")
    nlp64 === nothing && return
    nvar, ncon = nlp64.meta.nvar, nlp64.meta.ncon
    PTS = points(nlp64)
    y0, v0 = multipliers(nlp64)
    R64 = host_reference(nlp64, PTS, y0, v0)
    nlp64 = nothing
    dev = CUDABackend()

    gpu64 = build(Float64, dev, "gpu_f64")
    if gpu64 !== nothing
        for (pt, x) in PTS
            haskey(R64, pt) || continue
            inflight("gpu_f64 readings $pt")
            stage("gpu64_vs_cpu64", pt) do
                G = readings(gpu64, CuArray(x), CuArray(y0), CuArray(v0))
                emit_pair("gpu64_vs_cpu64", pt, R64[pt], G, Float64, nvar, ncon)
            end
        end
    end
    gpu64 = nothing; R64 = nothing
    GC.gc(); CUDA.reclaim()

    cpu32 = build(Float32, nothing, "cpu_f32")
    gpu32 = cpu32 === nothing ? nothing : build(Float32, dev, "gpu_f32")
    if gpu32 !== nothing
        for (pt, x) in PTS
            inflight("gpu_f32 readings $pt")
            stage("gpu32_vs_cpu32", pt) do
                H = readings(cpu32, Float32.(x), Float32.(y0), Float32.(v0))
                G = readings(gpu32, CuArray(Float32.(x)), CuArray(Float32.(y0)),
                             CuArray(Float32.(v0)))
                emit_pair("gpu32_vs_cpu32", pt, H, G, Float32, nvar, ncon)
            end
        end
    end
    cpu32 = nothing; gpu32 = nothing
    GC.gc(); CUDA.reclaim()
end

function run_cpu(prob)
    nlp64 = build(Float64, nothing, "cpu_f64")
    if nlp64 === nothing
        emitA(prob; callback = "PROBLEM", status = "BUILD_FAILED",
              note = "ExaCUTEstModel(name) failed; see cpu.tsv build row")
        return
    end
    nvar, ncon = nlp64.meta.nvar, nlp64.meta.ncon
    PTS = points(nlp64)
    y0, v0 = multipliers(nlp64)
    R64 = host_reference(nlp64, PTS, y0, v0; self = true)
    nlp32 = build(Float32, nothing, "cpu_f32")
    if nlp32 !== nothing
        for (pt, x) in PTS
            haskey(R64, pt) || continue
            inflight("cpu_f32 readings $pt")
            stage("cpu32_vs_cpu64", pt) do
                T32 = readings(nlp32, Float32.(x), Float32.(y0), Float32.(v0))
                emit_pair("cpu32_vs_cpu64", pt, R64[pt], T32, Float32, nvar, ncon)
            end
        end
    end
    nlp32 = nothing; R64 = nothing
    GC.gc()

    try
        if prob in NATIVE
            nlp64 = nothing; GC.gc()
            inflight("fortran native_verify")
            fortran_native(prob)
        else
            fortran_decoded(prob, nlp64)
        end
    catch e
        emitA(prob; callback = "error", status = "BAD", note = first(sprint(showerror, e), 400))
        emitA(prob; callback = "PROBLEM", status = "FAIL", note = "comparison raised")
    end
end

# --- task source --------------------------------------------------------------

"Next problem, or `nothing` when there is none left."
function next_problem()
    if MODE == "gpu"
        donef = joinpath(FR, "state", "done.$WID")
        done = isfile(donef) ? Set(readlines(donef)) : Set{String}()
        for p in readlines(joinpath(FR, ENV["CV_SHARD"]))
            isempty(p) || p in done || return p
        end
        return nothing
    end
    q = readlines(joinpath(FR, "state", "queue_cpu.txt"))
    nextf = joinpath(FR, "state", "queue_cpu.next")
    return withflock(nextf * ".lock") do
        k = isfile(nextf) ? parse(Int, strip(read(nextf, String))) : 1
        k > length(q) && return nothing
        write(nextf, string(k + 1))
        open(io -> println(io, "$(q[k])\t$WID\t$HOST"), joinpath(FR, "state", "claimed_cpu.txt"), "a")
        q[k]
    end
end

function main()
    if MODE == "gpu"
        dev = CUDA.device()
        @info "worker $WID on $HOST, $(CUDA.name(dev)), CUDA_VISIBLE_DEVICES=$(get(ENV, "CUDA_VISIBLE_DEVICES", "-"))"
    end
    while (prob = next_problem()) !== nothing
        CUR[] = prob
        inflight("start")
        t0 = time()
        arm(TMO)
        try
            MODE == "gpu" ? run_gpu(prob) : run_cpu(prob)
        catch e
            emit(; cmp = "worker", callback = "problem", status = "ERROR",
                 note = first(sprint(showerror, e), 400))
        end
        arm(0)
        emit(; cmp = "worker", callback = "PROBLEM", status = "DONE", secs = time() - t0)
        MODE == "gpu" && open(io -> println(io, prob), joinpath(FR, "state", "done.$WID"), "a")
        @info @sprintf("%s %.1f s", prob, time() - t0)
        write(joinpath(FR, "state", "inflight.$WID"), "")
    end
end

main()
