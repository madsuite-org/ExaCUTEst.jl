# ExaCUTEst.jl — the CUTEst test set as ExaModels, on CPU or GPU.
#
# Problems are decoded from their SIF files by ExaModelsSIF (a SIF front end
# for ExaModels) rather than hand-translated: every problem in the collection
# is available, and each one is a SIMD-abstracted ExaModel that can be built on
# a GPU backend. The API mirrors CUTEst.jl so existing code ports with a
# constructor rename.
module ExaCUTEst

using Artifacts, LazyArtifacts
using ExaModels
using ExaModelsSIF
using NLPModels
import JSON
import SpecialFunctions
import ForwardDiff

# Problems whose element/group functions are external Fortran: the decoder
# cannot translate them, so each has a native implementation in src/native/.
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
include("model.jl")

export ExaCUTEstModel,
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
    ba_residual,
    ba_oracle_model,
    sif_variable_names,
    sif_constraint_names,
    row_linearity

end # module
