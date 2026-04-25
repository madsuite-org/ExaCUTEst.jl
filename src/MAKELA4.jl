# MAKELA4: linear minmax problem in twenty variables (Makela 1990)
# 21 variables: X(1)..X(20), U (all free)
# minimize U
# subject to (L type: v <= 0, for i=1..20):
#   F(I):  -U + X(I) <= 0
#   MF(I): -U - X(I) <= 0
# Start: X(i) = i for i=1..10, X(i) = -i for i=11..20, U=0
# Note: CUTEst sifdecoder interleaves constraints: F(1),MF(1),F(2),MF(2),...
function makela4_model(; T = Float64, backend = nothing, kwargs...)
    start_vec = vcat(T[i for i in 1:10], T[-i for i in 11:20], T[0])
    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, 21; lvar = T(-Inf), uvar = T(Inf), start = start_vec)
    ExaModels.@add_obj(c, x[21])
    # Interleaved: F(i) and MF(i) for each i=1..20
    for i in 1:20
        ExaModels.@add_con(c, -x[21] + x[i]; lcon = T(-Inf), ucon = T(0))
        ExaModels.@add_con(c, -x[21] - x[i]; lcon = T(-Inf), ucon = T(0))
    end
    return ExaModels.ExaModel(c; kwargs...)
end
