# Precompiles the problems of the precompile set on Intel GPUs (oneAPI) (see src/precompile.jl).
# Needs a functional device at precompile time.
# oneAPI.jl skips its initialization while precompiling, so `functional` is false
# here and this workload stores nothing until that changes.
module ExaCUTEstoneAPIExt

import ExaCUTEst
import oneAPI
import PrecompileTools

PrecompileTools.@compile_workload begin
    if ExaCUTEst.PRECOMPILE && oneAPI.functional()
        ExaCUTEst._precompile_workload(oneAPI.oneAPIBackend())
    end
end

end
