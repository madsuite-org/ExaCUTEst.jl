function hs54_model(; T = Float64, backend = nothing, kwargs...)
    c = ExaModels.ExaCore(T; backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 6;
                       lvar = T[0.0, -10.0, 0.0, 0.0, -1.0, 0.0],
                       uvar = T[2e4, 10.0, 1e7, 20.0, 1.0, 2e8],
                       start = T[6e3, 1.5, 4e6, 2.0, 3e-3, 5e7])
    # GROUP TYPE NORMAL: phi(Q) = -exp(-Q/2) applied to the group value Q
    # Q = FACTOR*(E1+E2+F1) + E3+E4+E5+E6
    # E_i = ((x_i-mu_i)/sigma_i)^2, F1 = 2*rho*(x1-mu1)/sigma1*(x2-mu2)/sigma2
    # RHO=0.2, FACTOR=1/(1-RHO^2)=1/0.96
    ExaModels.@add_obj(c,
        -exp(-T(0.5) * (
            T(1.0 / 0.96) * (
                ((x[1] - T(1e4)) / T(8e3))^2 +
                (x[2] - T(1.0))^2 +
                T(0.4) * ((x[1] - T(1e4)) / T(8e3)) * (x[2] - T(1.0))
            ) +
            ((x[3] - T(2e6)) / T(7e6))^2 +
            ((x[4] - T(10.0)) / T(50.0))^2 +
            ((x[5] - T(1e-3)) / T(5e-2))^2 +
            ((x[6] - T(1e8)) / T(5e8))^2
        )))
    # CON1 (E): x1 + 4000*x2 = mu1 + 4000*mu2 + 0.2*sigma1 + 2000*sigma2 = 17600
    ExaModels.@add_con(c, x[1] + T(4e3)*x[2] - T(17600.0))
    return ExaModels.ExaModel(c; kwargs...)
end
