# Precompiles the problems of the precompile set on NVIDIA GPUs (CUDA) (see src/precompile.jl).
# Needs a functional device at precompile time.
# Below Julia 1.12 the stored GPU code does not load or does not help, so the
# workload is skipped there.
module ExaCUTEstCUDAExt

import ExaCUTEst
import CUDA
import PrecompileTools

PrecompileTools.@compile_workload begin
    if VERSION >= v"1.12" && ExaCUTEst.PRECOMPILE && CUDA.functional()
        ExaCUTEst._precompile_workload(CUDA.CUDABackend())
    end
end

end
