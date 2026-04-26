# HEART6LS: Dipole model of the heart (6-variable least-squares version)
# Source: Dennis, Gay, Vu (1983/1985), SIF input: Conn 1993, Toint 1994
# 6 variables (a,c,t,u,v,w), no constraints, 6 L2 least-squares groups
# Variables indexed: a=1, c=2, t=3, u=4, v=5, w=6
function heart6ls_model(; T = Float64, backend = nothing, kwargs...)
    sMx = T(-0.816)
    sMy = T(-0.017)
    sA  = T(-1.826)
    sB  = T(-0.754)
    sC  = T(-4.839)
    sD  = T(-3.259)
    sE  = T(-14.023)
    sF  = T(15.467)

    start = T[0, 0, 1, 1, 1, 1]
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6; lvar = T(-Inf), uvar = T(Inf), start = start)

    # G1: (t*a + u*(sMx-a) - v*c - w*(sMy-c) - sA)^2
    ExaModels.@add_obj(c,
        (x[3]*x[1] + x[4]*(sMx - x[1]) - x[5]*x[2] - x[6]*(sMy - x[2]) - sA)^2
    )
    # G2: (v*a + w*(sMx-a) + t*c + u*(sMy-c) - sB)^2
    ExaModels.@add_obj(c,
        (x[5]*x[1] + x[6]*(sMx - x[1]) + x[3]*x[2] + x[4]*(sMy - x[2]) - sB)^2
    )
    # G3: (a*(t^2-v^2) - 2*c*t*v + (sMx-a)*(u^2-w^2) - 2*(sMy-c)*u*w - sC)^2
    ExaModels.@add_obj(c,
        (x[1]*(x[3]^2 - x[5]^2) - T(2)*x[2]*x[3]*x[5] +
         (sMx - x[1])*(x[4]^2 - x[6]^2) - T(2)*(sMy - x[2])*x[4]*x[6] - sC)^2
    )
    # G4: (c*(t^2-v^2) + 2*a*t*v + (sMy-c)*(u^2-w^2) + 2*(sMx-a)*u*w - sD)^2
    ExaModels.@add_obj(c,
        (x[2]*(x[3]^2 - x[5]^2) + T(2)*x[1]*x[3]*x[5] +
         (sMy - x[2])*(x[4]^2 - x[6]^2) + T(2)*(sMx - x[1])*x[4]*x[6] - sD)^2
    )
    # G5: (a*t*(t^2-3v^2) + c*v*(v^2-3t^2) + (sMx-a)*u*(u^2-3w^2) + (sMy-c)*w*(w^2-3u^2) - sE)^2
    ExaModels.@add_obj(c,
        (x[1]*x[3]*(x[3]^2 - T(3)*x[5]^2) + x[2]*x[5]*(x[5]^2 - T(3)*x[3]^2) +
         (sMx - x[1])*x[4]*(x[4]^2 - T(3)*x[6]^2) +
         (sMy - x[2])*x[6]*(x[6]^2 - T(3)*x[4]^2) - sE)^2
    )
    # G6: (c*t*(t^2-3v^2) - a*v*(v^2-3t^2) + (sMy-c)*u*(u^2-3w^2) - (sMx-a)*w*(w^2-3u^2) - sF)^2
    ExaModels.@add_obj(c,
        (x[2]*x[3]*(x[3]^2 - T(3)*x[5]^2) - x[1]*x[5]*(x[5]^2 - T(3)*x[3]^2) +
         (sMy - x[2])*x[4]*(x[4]^2 - T(3)*x[6]^2) -
         (sMx - x[1])*x[6]*(x[6]^2 - T(3)*x[4]^2) - sF)^2
    )

    return ExaModels.ExaModel(c; kwargs...)
end
