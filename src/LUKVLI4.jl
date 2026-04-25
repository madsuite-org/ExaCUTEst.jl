# LUKVLI4: Inequality version of LUKVLE4 (L-type constraints ≤ 0)
# Source: Problem 5.4, Luksan & Vlcek (1999), TR 767
# N=10000 variables (default), N/2-1=4999 L-type constraints
# All variables free; start: X(4k+1)=1, X(4k+2..4k+4)=2
#
# CUTEst variable ordering: j(m)=N+1-m except j(N/2)=N/2, j(N/2+1)=N/2+1 (self-inverse)
# Identical to LUKVLE4 except constraints are ≤ 0 (lcon=-Inf, ucon=0)
function lukvli4_model(N = 10000; T = Float64, backend = nothing, kwargs...)
    @assert N % 4 == 0 "N must be divisible by 4"
    @assert N >= 16 "N must be >= 16"
    N2m1 = N ÷ 2 - 1
    half = N ÷ 2
    qtr  = N ÷ 4

    start = T[
        let m = (c == half || c == half + 1) ? c : N + 1 - c
            ((m - 1) % 4 == 0) ? T(1) : T(2)
        end
        for c in 1:N
    ]

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, N; lvar = T(-Inf), uvar = T(Inf), start = start)

    Imex1 = qtr - 1
    Imex2 = qtr
    Imex3 = qtr + 1
    Itri1 = half - 2
    Itri2 = half - 1

    ExaModels.@add_obj(c, (-x[N+1-2i] + exp(x[N+2-2i]))^4 for i in 1:(Imex2-1))
    ExaModels.@add_obj(c, (-x[half]   + exp(x[half+2]))^4)
    ExaModels.@add_obj(c, (-x[half-1] + exp(x[half+1]))^4)
    ExaModels.@add_obj(c, (-x[N+1-2i] + exp(x[N+2-2i]))^4 for i in (Imex3+1):N2m1)

    ExaModels.@add_obj(c, T(100)*(x[N+1-2i] - x[N-2i])^6 for i in 1:(Imex2-1))
    ExaModels.@add_obj(c, T(100)*(x[half] - x[half+1])^6)
    ExaModels.@add_obj(c, T(100)*(x[N+1-2i] - x[N-2i])^6 for i in (Imex2+1):N2m1)

    ExaModels.@add_obj(
        c,
        let u = x[N-2i] - x[N-1-2i]
            (u + tan(u) + T(6)*x[N-i] + T(8)*(x[N-i]^3 - x[N-i]*x[N+1-i]) -
             T(4)*x[N-1-i]^2 - T(2))^4
        end for i in 1:(Imex1-1)
    )
    ExaModels.@add_obj(c,
        let u = x[half+2] - x[half]
            (u + tan(u) + T(6)*x[half+qtr+1] + T(8)*(x[half+qtr+1]^3 - x[half+qtr+1]*x[half+qtr+2]) -
             T(4)*x[half+qtr]^2 - T(2))^4
        end
    )
    ExaModels.@add_obj(c,
        let u = x[half+1] - x[half-1]
            (u + tan(u) + T(6)*x[half+qtr] + T(8)*(x[half+qtr]^3 - x[half+qtr]*x[half+qtr+1]) -
             T(4)*x[half+qtr-1]^2 - T(2))^4
        end
    )
    ExaModels.@add_obj(
        c,
        let u = x[N-2i] - x[N-1-2i]
            (u + tan(u) + T(6)*x[N-i] + T(8)*(x[N-i]^3 - x[N-i]*x[N+1-i]) -
             T(4)*x[N-1-i]^2 - T(2))^4
        end for i in (Imex2+1):(Itri1-1)
    )
    ExaModels.@add_obj(c,
        let u = x[4] - x[3]
            (u + tan(u) + T(6)*x[half+2] + T(8)*(x[half+2]^3 - x[half+2]*x[half+3]) -
             T(4)*x[half]^2 - T(2))^4
        end
    )
    ExaModels.@add_obj(c,
        let u = x[2] - x[1]
            (u + tan(u) + T(6)*x[half] + T(8)*(x[half]^3 - x[half]*x[half+2]) -
             T(4)*x[half+1]^2 - T(2))^4
        end
    )

    ExaModels.@add_obj(c, x[N+2-2i]^8 for i in 1:(Imex3-1))
    ExaModels.@add_obj(c, x[half+1]^8)
    ExaModels.@add_obj(c, x[N+2-2i]^8 for i in (Imex3+1):N2m1)

    ExaModels.@add_obj(c, (x[N-1-2i] - T(1))^2 for i in 1:(Imex1-1))
    ExaModels.@add_obj(c, (x[half] - T(1))^2)
    ExaModels.@add_obj(c, (x[N-1-2i] - T(1))^2 for i in (Imex1+1):N2m1)

    ExaModels.@add_con(c,
        T(6)*x[half+1] + T(8)*(x[half+1]^3 - x[half+1]*x[half]) - T(4)*x[half-1]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(c,
        T(6)*x[half-1] + T(8)*(x[half-1]^3 - x[half-1]*x[half+1]) - T(4)*x[half-2]^2 - T(2);
        lcon = T(-Inf), ucon = T(0)
    )
    ExaModels.@add_con(c,
        T(6)*x[half+1-ci] + T(8)*(x[half+1-ci]^3 - x[half+1-ci]*x[half+2-ci]) -
        T(4)*x[half-ci]^2 - T(2)
        for ci in 3:N2m1;
        lcon = T(-Inf), ucon = T(0)
    )

    return ExaModels.ExaModel(c; kwargs...)
end
