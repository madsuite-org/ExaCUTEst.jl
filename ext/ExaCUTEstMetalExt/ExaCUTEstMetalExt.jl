# Precompiles the problems of the precompile set on Apple GPUs (Metal), in Float32 (see src/precompile.jl).
# Needs a functional device at precompile time.
# Below Julia 1.12 the stored GPU code does not load or does not help, so the
# workload is skipped there.
module ExaCUTEstMetalExt

import ExaCUTEst
import Metal
import PrecompileTools

PrecompileTools.@compile_workload begin
    if VERSION >= v"1.12" && ExaCUTEst.PRECOMPILE && Metal.functional()
        ExaCUTEst._precompile_workload(Metal.MetalBackend(); T = Float32)
    end
end

end
