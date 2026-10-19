# Switch a verification script to the test/gpu environment.
#
# CUDA and CUTEst.jl are not dependencies of ExaCUTEst (CUTEst.jl is a test
# dependency only), so the package's own environment cannot resolve `using
# CUDA` or `using CUTEst`. This directory is an environment that can, and the
# GPU and Fortran-reference verification scripts run in it.
#
# It is a workspace member of the package environment (`[workspace]` in the
# package's Project.toml), so the two share one Manifest.toml and every package
# common to them is at the same version. That is what keeps a GPU run from
# loading a different LLVM.jl or KernelAbstractions than a `--project=.` run:
# one resolver pass decides both. A CUDA installed in some other environment
# further along the load path would not be resolved against this one, and a
# CUDA from before LLVM.jl 10 fails to precompile against it.
#
# `LocalPreferences.toml` beside this file holds the CUDA runtime at a version
# that still supports compute capability 7.0; the default runtime has dropped
# it, and the Quadro GV100 is a 7.0 device.
#
# Switching environments here, rather than asking the caller to do it, is what
# makes `julia --project=. test/<name>_verify.jl` work.
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    @eval using Pkg
    Pkg.activate(@__DIR__; io = devnull)
end
