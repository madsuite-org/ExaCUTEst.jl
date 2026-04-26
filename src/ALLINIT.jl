# ALLINIT: "All in it" verification problem (Gould 1990)
# 4 variables: X1 free, X2≥1, X3∈[-1e10,1], X4=2 (fixed); no constraints
# Groups FT1-FT6 use TRIVIAL type (identity); FNT1-FNT6 use L2 type (square)
# Elements: SQR(X)=X², SQR2(Y,Z)=(Y+Z)², SINSQR(X)=sin²(X), PRODSQR(X,Y)=X²Y²
function allinit_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 4;
        lvar = T[T(-Inf), T(1), T(-1e10), T(2)],
        uvar = T[T(Inf), T(Inf), T(1), T(2)],
        start = T[T(0), T(0), T(0), T(0)])
    # FT2: trivial(X3 - 1)
    ExaModels.@add_obj(c, x[3] - T(1))
    # FT3: trivial(X1²)
    ExaModels.@add_obj(c, x[1]^2)
    # FT4: trivial(X2² + (X3+X4)²)
    ExaModels.@add_obj(c, x[2]^2 + (x[3]+x[4])^2)
    # FT5: trivial(X4 + sin²(X3) + X1²X2² - 3)
    ExaModels.@add_obj(c, x[4] + sin(x[3])^2 + x[1]^2*x[2]^2 - T(3))
    # FT6: trivial(sin²(X3))
    ExaModels.@add_obj(c, sin(x[3])^2)
    # FNT2: L2((X4-1)²) = (X4-1)²
    ExaModels.@add_obj(c, (x[4] - T(1))^2)
    # FNT3: L2(X2²) = X2⁴
    ExaModels.@add_obj(c, x[2]^4)
    # FNT4: L2(X3² + (X4+X1)²) = (X3² + (X4+X1)²)²
    ExaModels.@add_obj(c, (x[3]^2 + (x[4]+x[1])^2)^2)
    # FNT5: L2(X1 + sin²(X4) + X2²X3² - 4) = (X1 + sin²(X4) + X2²X3² - 4)²
    ExaModels.@add_obj(c, (x[1] + sin(x[4])^2 + x[2]^2*x[3]^2 - T(4))^2)
    # FNT6: L2(sin²(X4)) = sin⁴(X4)
    ExaModels.@add_obj(c, sin(x[4])^4)
    return ExaModels.ExaModel(c; kwargs...)
end
