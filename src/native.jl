# native.jl — hand-written ExaModels for problems the decoder cannot reach.
#
# A problem lands here only when BOTH hold: its SIF file uses external Fortran
# (so sif_decode refuses it), and the Fortran computes something Julia can
# express exactly.
#
# TWO SHAPES live here. HS68/HS69 are small enough to write as ExaModels
# expressions directly. The BA-L* family is NOT: its reprojection chain is
# ~30 operations deep, and built as an expression it did not finish compiling
# in an hour on a 57-variable problem, because ExaModels nests a type
# parameter per operation. That family is instead a GENERIC JULIA FUNCTION
# interfaced as an external oracle, with derivatives by AD over it (Sungho
# 2026-10-16: "it should be a generic julia function" ... "this generic julia
# function can be interfaced with ExaModels as an external oracle"). Prefer
# that shape for anything non-trivial: it compiles like ordinary Julia, it is
# generic over the element type for free, and being allocation- and
# branch-free it is callable from a GPU kernel. The gain over the oracle tier is real -- a native model is
# a SIMD-abstracted ExaModel that runs on a GPU, where an oracle is a CPU
# Fortran callback whose Hessian costs one hprod per column.
#
# Each entry must be validated against a CUTEstModel of the same problem
# before being added (see test/runtests.jl, "native models"); a hand-written
# model that merely looks right is worse than an oracle, because it is wrong
# silently. The legacy 0.1 port contained three candidates and only two
# survived that check.

include("native/HS68.jl")
include("native/HS69.jl")
include("native/BA.jl")

"""
    NATIVE_MODELS :: Dict{String,Function}

External-Fortran problems that have a verified native ExaModels
implementation, by problem name. `ExaCUTEstModel` prefers these over the
oracle tier, so they accept a `backend` and run on a GPU.
"""
const NATIVE_MODELS = Dict{String,Function}(
    "HS68" => hs68_model,
    "HS69" => hs69_model,
    (p => (; kwargs...) -> ba_oracle_model(p; kwargs...)
     for p in ("BA-L1", "BA-L1LS", "BA-L16", "BA-L16LS", "BA-L21", "BA-L21LS",
               "BA-L49", "BA-L49LS", "BA-L52", "BA-L52LS", "BA-L73", "BA-L73LS"))...,
)

"""
    native_problems() -> Vector{String}

Problems served by a verified hand-written ExaModel rather than by decoding
or by the oracle tier.
"""
native_problems() = sort(collect(keys(NATIVE_MODELS)))

"""
    oracle_tier_problems() -> Vector{String}

External-Fortran problems that still go through the CUTEst-backed oracle
tier: [`external_fortran_problems`](@ref) minus [`native_problems`](@ref).
These are CPU-only and refuse a `backend`.
"""
oracle_tier_problems() =
    sort(setdiff(external_fortran_problems(), native_problems()))
