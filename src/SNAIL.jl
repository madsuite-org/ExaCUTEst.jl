# SNAIL: 2D spiraling valley function
# Source: J. Engels (private communication), Ph. Toint
# F(x,y) = U * V where:
#   U = R^2 / (1 + R^2),  R^2 = x^2 + y^2
#   V = 1 + A*R - R*C,    A = 0.5*(CUP+CLOW), B = 0.5*(CUP-CLOW),
#                          C = B*cos(R - theta),  theta = atan2(y,x)
# CLOW=1, CUP=2: A=1.5, B=0.5
# Bounds: free; start: x=10, y=10
function snail_model(; T = Float64, backend = nothing, kwargs...)
    CL = T(1.0)
    CU = T(2.0)
    A  = T(0.5) * (CU + CL)   # 1.5
    B  = T(0.5) * (CU - CL)   # 0.5
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 2; lvar = T(-Inf), uvar = T(Inf), start = T[10.0, 10.0])
    ExaModels.@add_obj(c,
        let R2 = x[1]^2 + x[2]^2,
            R  = sqrt(R2),
            theta = atan(x[2], x[1]),
            U  = R2 / (T(1) + R2),
            C  = B * cos(R - theta),
            V  = T(1) + A * R - R * C
            U * V
        end
    )
    return ExaModels.ExaModel(c; kwargs...)
end
