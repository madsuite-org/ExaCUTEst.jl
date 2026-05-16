# ELEC: Electron equilibrium on a unit sphere, NP=200 default
# Minimize sum_{I<J} 1/||(X(I),Y(I),Z(I)) - (X(J),Y(J),Z(J))||
# Subject to: X(I)^2 + Y(I)^2 + Z(I)^2 = 1 for each I
# Variables: X(I) at 3*(I-1)+1, Y(I) at 3*(I-1)+2, Z(I) at 3*(I-1)+3
# Start: quasi-uniform distribution theta(I) = 2pi*I/NP, phi(I) = pi*(I-1)/NP
function elec_model(NP = 200; T = Float64, backend = nothing, kwargs...)
    X_idx(i) = 3 * (i - 1) + 1
    Y_idx(i) = 3 * (i - 1) + 2
    Z_idx(i) = 3 * (i - 1) + 3
    NV = 3 * NP

    invN = T(1) / T(NP)
    pi_val = T(4) * atan(T(1))

    x0 = zeros(T, NV)
    for i in 1:NP
        theta = T(2) * pi_val * T(i) * invN
        phi   = pi_val * T(i - 1) * invN
        x0[X_idx(i)] = cos(theta) * sin(phi)
        x0[Y_idx(i)] = sin(theta) * sin(phi)
        x0[Z_idx(i)] = cos(phi)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # Objective: sum_{I<J} 1/sqrt((xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2)
    pair_data = [(
        xi = X_idx(i), yi = Y_idx(i), zi = Z_idx(i),
        xj = X_idx(j), yj = Y_idx(j), zj = Z_idx(j),
    ) for i in 1:(NP - 1) for j in (i + 1):NP]

    ExaModels.@add_obj(c,
        T(1) / sqrt((x[d.xi] - x[d.xj])^2 + (x[d.yi] - x[d.yj])^2 + (x[d.zi] - x[d.zj])^2)
        for d in pair_data)

    # Ball constraints: X(I)^2 + Y(I)^2 + Z(I)^2 - 1 = 0
    ExaModels.@add_con(c,
        x[X_idx(i)]^2 + x[Y_idx(i)]^2 + x[Z_idx(i)]^2 - T(1)
        for i in 1:NP)

    return ExaModels.ExaModel(c; kwargs...)
end
