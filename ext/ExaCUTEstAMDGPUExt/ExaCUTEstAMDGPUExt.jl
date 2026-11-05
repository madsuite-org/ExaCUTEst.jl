# Precompiles the problems of the precompile set on AMD GPUs (AMDGPU) (see src/precompile.jl).
# Needs a functional device at precompile time.
module ExaCUTEstAMDGPUExt

import ExaCUTEst
import AMDGPU
import PrecompileTools

PrecompileTools.@compile_workload begin
    if ExaCUTEst.PRECOMPILE && AMDGPU.functional()
        ExaCUTEst._precompile_workload(AMDGPU.ROCBackend())
    end
end

end
