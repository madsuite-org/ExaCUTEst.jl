# HEART8LS: Dipole model of the heart (8-variable least-squares version)
# Source: Dennis, Gay, Vu (1983/1985), SIF input: Conn 1993, correction Shott 1995
# 8 variables (a,b,c,d,t,u,v,w), no constraints, 8 L2 least-squares groups
# Variables indexed: a=1, b=2, c=3, d=4, t=5, u=6, v=7, w=8
function heart8ls_model(; T = Float64, backend = nothing, kwargs...)
    sMx = T(-0.69)
    sMy = T(-0.044)
    sA  = T(-1.57)
    sB  = T(-1.31)
    sC  = T(-2.65)
    sD  = T(2.0)
    sE  = T(-12.6)
    sF  = T(9.48)

    start = T[0, 1, 0, 1, 1, 1, 1, 1]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 8; lvar = T(-Inf), uvar = T(Inf), start = start)

    # G1: (a + b - sMx)^2
    ExaModels.@add_obj(c, (x[1] + x[2] - sMx)^2)
    # G2: (c + d - sMy)^2
    ExaModels.@add_obj(c, (x[3] + x[4] - sMy)^2)
    # G3: (t*a + u*b - v*c - w*d - sA)^2
    ExaModels.@add_obj(c, (x[5]*x[1] + x[6]*x[2] - x[7]*x[3] - x[8]*x[4] - sA)^2)
    # G4: (v*a + w*b + t*c + u*d - sB)^2
    ExaModels.@add_obj(c, (x[7]*x[1] + x[8]*x[2] + x[5]*x[3] + x[6]*x[4] - sB)^2)
    # G5: (a*(t^2-v^2) - 2*c*t*v + b*(u^2-w^2) - 2*d*u*w - sC)^2
    ExaModels.@add_obj(c,
        (x[1]*(x[5]^2 - x[7]^2) - T(2)*x[3]*x[5]*x[7] +
         x[2]*(x[6]^2 - x[8]^2) - T(2)*x[4]*x[6]*x[8] - sC)^2
    )
    # G6: (c*(t^2-v^2) + 2*a*t*v + d*(u^2-w^2) + 2*b*u*w - sD)^2
    ExaModels.@add_obj(c,
        (x[3]*(x[5]^2 - x[7]^2) + T(2)*x[1]*x[5]*x[7] +
         x[4]*(x[6]^2 - x[8]^2) + T(2)*x[2]*x[6]*x[8] - sD)^2
    )
    # G7: (a*t*(t^2-3v^2) + c*v*(v^2-3t^2) + b*u*(u^2-3w^2) + d*w*(w^2-3u^2) - sE)^2
    ExaModels.@add_obj(c,
        (x[1]*x[5]*(x[5]^2 - T(3)*x[7]^2) + x[3]*x[7]*(x[7]^2 - T(3)*x[5]^2) +
         x[2]*x[6]*(x[6]^2 - T(3)*x[8]^2) + x[4]*x[8]*(x[8]^2 - T(3)*x[6]^2) - sE)^2
    )
    # G8: (c*t*(t^2-3v^2) - a*v*(v^2-3t^2) + d*u*(u^2-3w^2) - b*w*(w^2-3u^2) - sF)^2
    ExaModels.@add_obj(c,
        (x[3]*x[5]*(x[5]^2 - T(3)*x[7]^2) - x[1]*x[7]*(x[7]^2 - T(3)*x[5]^2) +
         x[4]*x[6]*(x[6]^2 - T(3)*x[8]^2) - x[2]*x[8]*(x[8]^2 - T(3)*x[6]^2) - sF)^2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
