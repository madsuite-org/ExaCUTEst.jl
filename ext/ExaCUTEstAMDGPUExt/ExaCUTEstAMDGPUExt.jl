# Precompiles the problems of the precompile set on AMD GPUs (AMDGPU) (see src/precompile.jl).
# Needs a functional device at precompile time.
# Below Julia 1.12 the stored GPU code does not load or does not help, so the
# workload is skipped there.
module ExaCUTEstAMDGPUExt

import ExaCUTEst
import AMDGPU
import PrecompileTools

PrecompileTools.@compile_workload begin
    if VERSION >= v"1.12" && ExaCUTEst.PRECOMPILE && AMDGPU.functional()
        ExaCUTEst._precompile_workload(AMDGPU.ROCBackend())
    end
end

end
