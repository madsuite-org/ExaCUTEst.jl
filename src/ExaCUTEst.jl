# ExaCUTEst.jl — the CUTEst test set as ExaModels, on CPU or GPU.
#
# Problems are decoded from their SIF files by ExaModelsSIF (a SIF front end
# for ExaModels) rather than hand-translated: every problem in the collection
# is available, and each one is a SIMD-abstracted ExaModel that can be built on
# a GPU backend. The API mirrors CUTEst.jl so existing code ports with a
# constructor rename.
#
# History: versions through 0.1 shipped one hand-written Julia file per
# problem. That is gone as of 0.2 -- the decoder covers all 1548 problems and
# is validated against the Fortran reference, so per-problem source would only
# be a second thing to keep correct.
module ExaCUTEst

using Artifacts, LazyArtifacts
using ExaModels
using ExaModelsSIF
using NLPModels
import CUTEst
import JSON
import SpecialFunctions
import ForwardDiff

# Problems whose element/group functions are external Fortran: the decoder
# cannot translate them, so they go through the CUTEst-backed oracle tier.
const _EXTERNAL_FORTRAN = String[
    "AKIVA",
    "BA-L1",
    "BA-L16",
    "BA-L16LS",
    "BA-L1LS",
    "BA-L21",
    "BA-L21LS",
    "BA-L49",
    "BA-L49LS",
    "BA-L52",
    "BA-L52LS",
    "BA-L73",
    "BA-L73LS",
    "BLEACHNG",
    "CAR2",
    "HIELOW",
    "HS67",
    "HS68",
    "HS69",
    "HS85",
    "HS88",
    "HS89",
    "HS90",
    "HS91",
    "HS92",
    "JIMACK",
    "ORBIT2",
    "PARKCH",
    "RAYBENDS",
    "SARO",
    "SAROMM",
    "STRATEC",
    "VANDANIUMS",
    "VANDANMSLS",
]

include("collection.jl")
include("classification.jl")
include("native.jl")
include("oracle.jl")
include("model.jl")

export ExaCUTEstModel,
    cutest_oracle_model,
    clear_oracle_cache,
    set_mastsif,
    mastsif,
    list_sif_problems,
    generator_sif_problems,
    sif_problem_path,
    select_sif_problems,
    classification,
    parse_classification,
    is_truly_unconstrained,
    external_fortran_problems,
    native_problems,
    oracle_tier_problems,
    ba_residual,
    ba_oracle_model,
    sif_variable_names,
    sif_constraint_names,
    row_linearity

end # module
