# native.jl — hand-written ExaModels for problems the decoder cannot reach.
#
# A problem lands here only when BOTH hold: its SIF file uses external Fortran
# (so sif_decode refuses it), and the Fortran computes something Julia can
# express exactly. The gain over the oracle tier is real -- a native model is
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

"""
    NATIVE_MODELS :: Dict{String,Function}

External-Fortran problems that have a verified native ExaModels
implementation, by problem name. `ExaCUTEstModel` prefers these over the
oracle tier, so they accept a `backend` and run on a GPU.
"""
const NATIVE_MODELS = Dict{String,Function}(
    "HS68" => hs68_model,
    "HS69" => hs69_model,
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
