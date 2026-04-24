module ExaCUTEst

using ExaModels

include("ROSENBR.jl")
include("HS3.jl")
include("HS4.jl")
include("HS5.jl")
include("HS9.jl")
include("HS21.jl")
include("TAME.jl")
include("ALSOTAME.jl")
include("DEGDIAG.jl")
include("ZANGWIL3.jl")
include("BOOTH.jl")
include("HIMMELBC.jl")
include("HIMMELBD.jl")
include("RSNBRNE.jl")
include("BT1.jl")
include("ZECEVIC2.jl")

export rosenbr_model, hs3_model, hs4_model, hs5_model, hs9_model, hs21_model,
       tame_model, alsotame_model, degdiag_model,
       zangwil3_model, booth_model, himmelbc_model, himmelbd_model,
       rsnbrne_model, bt1_model, zecevic2_model

end # module ExaCUTEst
