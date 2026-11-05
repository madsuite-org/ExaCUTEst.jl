# Precompiles the problems of the precompile set on NVIDIA GPUs (CUDA) (see src/precompile.jl).
# Needs a functional device at precompile time.
module ExaCUTEstCUDAExt

import ExaCUTEst
import CUDA
import PrecompileTools

PrecompileTools.@compile_workload begin
    if ExaCUTEst.PRECOMPILE && CUDA.functional()
        ExaCUTEst._precompile_workload(CUDA.CUDABackend())
    end
end

end
