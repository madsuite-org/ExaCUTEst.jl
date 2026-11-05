# Precompiles the problems of the precompile set on Apple GPUs (Metal), in Float32 (see src/precompile.jl).
# Needs a functional device at precompile time.
module ExaCUTEstMetalExt

import ExaCUTEst
import Metal
import PrecompileTools

PrecompileTools.@compile_workload begin
    if ExaCUTEst.PRECOMPILE && Metal.functional()
        ExaCUTEst._precompile_workload(Metal.MetalBackend(); T = Float32)
    end
end

end
