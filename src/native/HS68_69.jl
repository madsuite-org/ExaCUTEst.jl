# HS68 and HS69: cost-optimal inspection plans, problems 68 and 69 in Hock &
# Schittkowski (1981). Classification OOR2-MN-4-2.
#
# WHY THESE ARE NATIVE ExaModels RATHER THAN ORACLES. The appended Fortran
# computes the standard normal CDF, which Julia has exactly through `erf`, so
# nothing here needs an external callback: the model is an ordinary SIMD
# ExaModel, accepts a `backend`, and runs on a GPU. That is the only reason to
# hand-write a model in this package -- where the external routine is deep or
# depends on the variables, the port is a generic Julia function behind an
# oracle instead (see src/native/BA.jl).
#
#     minimize   AN/X1 - X4 (B (e^X1 - 1) - X3) / ((e^X1 - 1 + X4) X1)
#     subject to X3 - 2 PHI(-X2)                        = 0
#                X4 - PHI(DROOTN - X2) - PHI(-DROOTN - X2) = 0
#
# with PHI(t) = (1 + erf(t/sqrt 2))/2, AN = A NN and DROOTN = D sqrt(NN).
# Bounds X1 in [1e-4, 100], X2 in [0, 100], X3 and X4 in [0, 2]; start all 1.
#
#   HS68: A = 1e-4, B = 1,    D = 1, NN = 24
#   HS69: A = 0.1,  B = 1000, D = 1, NN = 4
#
# VARIABLE ORDER IS SIF DECLARATION ORDER -- x[1] = X1, x[2] = X2, x[3] = X3,
# x[4] = X4 -- as in every other model this package builds, and as
# `sif_variable_names` reports. A `CUTEstModel` of either problem puts the
# nonlinear variables first and so hands back X2 at position 1; comparing the
# two models therefore requires the name permutation, and indexing one model's
# vector with the other's positions silently reads the wrong variable.
function _hs68_69_model(::Type{T}, A, B, NN; backend, prod = :auto, kwargs...) where {T}
    AN = T(A) * T(NN)
    DROOTN = sqrt(T(NN))
    sq2 = sqrt(T(2))
    phi(t) = (T(1) + SpecialFunctions.erf(t / sq2)) / T(2)

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar = T[1e-4, 0.0, 0.0, 0.0],
        uvar = T[100.0, 100.0, 2.0, 2.0],
        start = T[1.0, 1.0, 1.0, 1.0])

    ExaModels.@add_obj(c, let
        X1 = x[1]; X3 = x[3]; X4 = x[4]
        E = exp(X1)
        AN / X1 - X4 * (T(B) * (E - T(1)) - X3) / ((E - T(1) + X4) * X1)
    end)

    ExaModels.@add_con(c, let
        X2 = x[2]; X3 = x[3]
        X3 - T(2) * phi(-X2)
    end)

    ExaModels.@add_con(c, let
        X2 = x[2]; X4 = x[4]
        X4 - phi(DROOTN - X2) - phi(-DROOTN - X2)
    end)

    return ExaModels.ExaModel(c; prod = ExaModelsSIF.prodflag(prod, c), kwargs...)
end

"""
    hs68_model(; T = Float64, backend = nothing, prod = :auto, kwargs...) -> ExaModel

HS68 as a native ExaModel: its external Fortran is the standard normal CDF,
which `erf` supplies exactly, so no oracle is needed and the model accepts a
`backend`.
"""
hs68_model(; T = Float64, backend = nothing, prod = :auto, kwargs...) =
    _hs68_69_model(T, 1e-4, 1.0, 24.0; backend, prod, kwargs...)

"""
    hs69_model(; T = Float64, backend = nothing, prod = :auto, kwargs...) -> ExaModel

HS69, the same model as [`hs68_model`](@ref) with A = 0.1, B = 1000, NN = 4.
"""
hs69_model(; T = Float64, backend = nothing, prod = :auto, kwargs...) =
    _hs68_69_model(T, 0.1, 1000.0, 4.0; backend, prod, kwargs...)
