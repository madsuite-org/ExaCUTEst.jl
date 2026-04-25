# MAKELA3: nonlinear minmax problem in twenty variables (Makela 1990)
# 21 variables: X(1)..X(20), U (all free)
# minimize U
# subject to (L type: group value >= 0, for i=1..20):
#   F(I): -U + X(I)^2 >= 0
# Start: X(i) = i for i=1..10, X(i) = -i for i=11..20, U=0
function makela3_model(; T = Float64, backend = nothing, kwargs...)
    start_vec = vcat(T[i for i in 1:10], T[-i for i in 11:20], T[0])
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 21; lvar = T(-Inf), uvar = T(Inf), start = start_vec)
    ExaModels.@add_obj(c, x[21])
    ExaModels.@add_con(c, -x[21] + x[i]^2 for i in 1:20; lcon = T(-Inf), ucon = T(0))
    return ExaModels.ExaModel(c; kwargs...)
end
