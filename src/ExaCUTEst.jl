module ExaCUTEst

using ExaModels

include("ROSENBR.jl")
include("HS5.jl")
include("TAME.jl")
include("ALSOTAME.jl")
include("DEGDIAG.jl")

export rosenbr_model, hs5_model, tame_model, alsotame_model, degdiag_model

end # module ExaCUTEst
