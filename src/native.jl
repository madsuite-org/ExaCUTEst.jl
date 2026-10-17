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

include("native/HS68_69.jl")
include("native/BA.jl")
# after BA.jl: the kernel path reuses ba_observations/ba_residual from it
include("native/BA_gpu.jl")
include("native/VANDANIUM.jl")
include("native/HS88_92.jl")
include("native/LOGIT.jl")
# after LOGIT.jl: the kernel path reuses HieLoWData and hielow_data from it,
# and serves PARKCH and STRATEC as well as AKIVA and HIELOW
include("native/LOGIT_gpu.jl")
include("native/SARO.jl")
# after SARO.jl: the kernel path reuses saro_stage_values, saro_stage_data,
# saro_tables, saro_stages and SARO_EVARS from it
include("native/SARO_gpu.jl")
include("native/HS67_85.jl")
# after LOGIT.jl: PARKCH/STRATEC reuse hielow_parse/hielow_loglik, because their
# SIFs append the SAME SUBROUTINE HIELOW (204 lines byte-identical once
# comments and DATA are stripped)
include("native/PARKCH_STRATEC.jl")
include("native/JIMACK.jl")
# after JIMACK.jl: the kernel path reuses JimackElement, jimack_elements and
# the element value and derivative functions from it
include("native/JIMACK_gpu.jl")
# CAR2/ORBIT2 share one file: their SIFs append the SAME `COLLOC` driver, and
# only the FSYS right-hand side differs (see the header there)
include("native/COLLOC.jl")
# after COLLOC.jl: the kernel path reuses `colloc`, the two FSYS structs,
# CollocElem/collocvar and the shared decode and SIMD-core helpers
include("native/COLLOC_gpu.jl")
include("native/RAYBENDS.jl")
# after RAYBENDS.jl: the kernel path reuses raybends_time/raybends_grad/
# raybends_hvp, bleachng_conc and its derivative sweeps, the decode helpers
# and _rb_hess_pattern
include("native/RAYBENDS_gpu.jl")

# The BA-L* family has two builders and the element type picks between them.
#
# `ba_kernel_model` is the one to want: KernelAbstractions kernels over the
# same `ba_residual`, so it serves a device as well as the host and a
# CPU-versus-GPU comparison tests the device rather than two unrelated
# implementations. It is also faster on the host at every size and callback
# measured, single-threaded -- 9.6x on BA-L1's `cons`, 22.2x on BA-L52's,
# 9.4x on BA-L52's `hess_coord`.
#
# It cannot do a non-bits element type. The objective gradient and both
# Hessian-vector products scatter-accumulate with `KernelAbstractions.@atomic`,
# which has no implementation for one: `T = BigFloat` raises
# `ArgumentError: unsupported atomic type BigFloat` even with
# `backend = nothing`, since the kernels run through KA's CPU backend either
# way. BigFloat is how a Float64 result gets arbitrated, so routing everything
# through the kernels would cost the instrument rather than a feature.
# `ba_oracle_model`'s plain host loops need no atomics and serve that case;
# they are also what was verified against the Fortran directly.
#
# `isbitstype` is necessary rather than sufficient -- a bits type KA happens
# to have no atomic for raises the same error, which names the type.
_ba_model(p::AbstractString; T = Float64, kwargs...) =
    isbitstype(T) ? ba_kernel_model(p; T, kwargs...) :
                    ba_oracle_model(p; T, kwargs...)

# JIMACK likewise has two builders that the element type picks between, and for
# the same reason: `jimack_kernel_model`'s objective gradient and its
# Hessian-vector product scatter-accumulate with `KernelAbstractions.@atomic`,
# which has no implementation for a non-bits type. Its value and Hessian
# COORDINATE callbacks do work at BigFloat -- they need no atomics -- so the
# routing is on the whole model rather than per callback. `jimack_model`'s host
# loops serve the non-bits case and are what was verified against the Fortran.
#
# The two declare DIFFERENT Hessian sparsity: `jimack_kernel_model` is
# blockwise with duplicates (259200 entries, a disjoint write per thread),
# `jimack_model` the deduplicated union (118824, which is also CUTEst's
# `nnzh`). Both assemble to the same matrix, so `nnzh` here depends on the
# element type while the Hessian does not. The duplicated pattern is what lets
# the device `hess!` be a pure write with no atomics, which is why it is
# bitwise reproducible across backends where a deduplicated one would not be.
_jimack_model(; T = Float64, kwargs...) =
    isbitstype(T) ? jimack_kernel_model(; T, kwargs...) :
                    jimack_model(; T, kwargs...)

# AKIVA, HIELOW, PARKCH and STRATEC are registered to their kernel builders
# too, and need no element-type gate for the same reason the HS kernels do not:
# the reduction over the estimation sample is two-stage -- one disjoint partial
# per (slot, record) thread, then a sequential sum per slot in increasing
# record order -- so no kernel uses `KernelAbstractions.@atomic`, the result is
# bitwise reproducible across backends, and all three element types build and
# evaluate through them. The kernel and host builders declare the SAME Hessian
# sparsity, the dense lower triangle of the coefficient block, so `nnzh` does
# not depend on which is registered, unlike JIMACK.
#
# CAR2, ORBIT2, RAYBENDS, BLEACHNG, SARO and SAROMM likewise need no gate,
# by a third
# scatter strategy: every accumulating callback is a disjoint per-item write
# into scratch followed by a gather, one thread per OUTPUT SLOT, over a table
# fixed at build time -- so the summation order is identical on every backend
# and no kernel uses `KernelAbstractions.@atomic`. CAR2 and ORBIT2 declare the
# blockwise pattern with duplicates and RAYBENDS and BLEACHNG the deduplicated
# union, matching their host builders entry for entry either way, so `nnzh`
# does not depend on which builder is registered.
#
# SARO and SAROMM need only one gather, for `hvp!`: no two of their 365 stages
# share a variable, so the blockwise Hessian pattern carries no duplicate
# coordinate and coincides with the deduplicated union at 24090 entries, which
# is what both builders declare. Their kernels are one thread per (stage,
# lower-triangle PAIR) rather than per column -- six times the arithmetic of
# the per-column shape and seven times faster, because at this depth spill
# traffic dominates: 5.5 KB per thread against 86 KB per column and 508 KB for
# a plain nested Hessian, which does not fit at all.
#
# HS67, HS85 and HS88-HS92 each have two builders as well, and the registered
# one is the kernel builder in every case: it accepts a `backend` and agrees
# with the host loops to 9e-16 over every callback at two points
# (test/hs_gpu_verify.jl). They need no `isbitstype` gate, unlike the two
# above, because every item they evaluate is written by a single thread, so no
# kernel uses `KernelAbstractions.@atomic` and all three precisions build and
# evaluate through them.

"""
    NATIVE_MODELS :: Dict{String,Function}

External-Fortran problems that have a verified native ExaModels
implementation, by problem name. `ExaCUTEstModel` prefers these over the
oracle tier, so they accept a `backend` and run on a GPU.
"""
const NATIVE_MODELS = Dict{String,Function}(
    "HS68" => hs68_model,
    "HS69" => hs69_model,
    (p => (; kwargs...) -> _ba_model(p; kwargs...)
     for p in ("BA-L1", "BA-L1LS", "BA-L16", "BA-L16LS", "BA-L21", "BA-L21LS",
               "BA-L49", "BA-L49LS", "BA-L52", "BA-L52LS", "BA-L73", "BA-L73LS"))...,
    (p => (; kwargs...) -> vandanium_model(p; kwargs...)
     for p in ("VANDANIUMS", "VANDANMSLS"))...,
    "HS88" => hs88_kernel_model,
    "HS89" => hs89_kernel_model,
    "HS90" => hs90_kernel_model,
    "HS91" => hs91_kernel_model,
    "HS92" => hs92_kernel_model,
    "AKIVA" => akiva_kernel_model,
    "HIELOW" => hielow_kernel_model,
    "SARO" => saro_kernel_model,
    "SAROMM" => saromm_kernel_model,
    "HS67" => hs67_kernel_model,
    "HS85" => hs85_kernel_model,
    "PARKCH" => parkch_kernel_model,
    "STRATEC" => stratec_kernel_model,
    "JIMACK" => _jimack_model,
    "CAR2" => car2_kernel_model,
    "ORBIT2" => orbit2_kernel_model,
    "RAYBENDS" => raybends_kernel_model,
    "BLEACHNG" => bleachng_kernel_model,
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
