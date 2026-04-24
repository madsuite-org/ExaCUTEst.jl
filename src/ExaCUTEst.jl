module ExaCUTEst

using ExaModels

include("ROSENBR.jl")
include("HS1.jl")
include("HS2.jl")
include("HS3.jl")
include("HS4.jl")
include("HS5.jl")
include("HS6.jl")
include("HS7.jl")
include("HS8.jl")
include("HS9.jl")
include("HS10.jl")
include("HS11.jl")
include("HS12.jl")
include("HS13.jl")
include("HS14.jl")
include("HS15.jl")
include("HS16.jl")
include("HS17.jl")
include("HS18.jl")
include("HS19.jl")
include("HS20.jl")
include("HS21.jl")
include("HS22.jl")
include("HS23.jl")
include("HS24.jl")
include("HS26.jl")
include("HS27.jl")
include("HS28.jl")
include("HS29.jl")
include("HS30.jl")
include("HS31.jl")
include("HS32.jl")
include("HS33.jl")
include("HS34.jl")
include("HS35.jl")
include("HS36.jl")
include("HS37.jl")
include("HS38.jl")
include("TAME.jl")
include("ALSOTAME.jl")
include("DEGDIAG.jl")
include("ZANGWIL3.jl")
include("BOOTH.jl")
include("HIMMELBC.jl")
include("HIMMELBD.jl")
include("HIMMELBE.jl")
include("RSNBRNE.jl")
include("BT1.jl")
include("ZECEVIC2.jl")
include("EXPFIT.jl")
include("LSQFIT.jl")

export rosenbr_model,
       hs1_model, hs2_model, hs3_model, hs4_model, hs5_model,
       hs6_model, hs7_model, hs8_model, hs9_model, hs10_model,
       hs11_model, hs12_model, hs13_model, hs14_model, hs15_model,
       hs16_model, hs17_model, hs18_model, hs19_model, hs20_model,
       hs21_model, hs22_model, hs23_model, hs24_model,
       hs26_model, hs27_model, hs28_model, hs29_model, hs30_model,
       hs31_model, hs32_model, hs33_model, hs34_model,
       hs35_model, hs36_model, hs37_model, hs38_model,
       tame_model, alsotame_model, degdiag_model,
       zangwil3_model, booth_model, himmelbc_model, himmelbd_model, himmelbe_model,
       rsnbrne_model, bt1_model, zecevic2_model,
       expfit_model, lsqfit_model

end # module ExaCUTEst
