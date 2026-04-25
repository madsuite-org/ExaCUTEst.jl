# CHAIN: Minimal potential energy of a hanging chain
# NH=400 intervals → 2*(NH+1)=802 variables, NH+1=401 constraints (NH derivative + 1 length)
# Variables x[1..NH+1]=X(0..NH), x[NH+2..2NH+2]=U(0..NH)
# Fixed: X(0)=A=1.0, X(NH)=B=3.0; all others free
# Objective (trapezoidal PE):
#   PE = H/2*X(0)*sqrt(1+U(0)^2) + H*sum_{I=1}^{NH-1} X(I)*sqrt(1+U(I)^2) + H/2*X(NH)*sqrt(1+U(NH)^2)
# D(I) constraints: X(I)-X(I+1)+H/2*(U(I)+U(I+1))=0  for I=0..NH-1
# LENGTH constraint: H/2*sqrt(1+U(0)^2) + H*sum_{I=1}^{NH-1} sqrt(1+U(I)^2) + H/2*sqrt(1+U(NH)^2) - L = 0
function chain_model(NH = 400; T = Float64, backend = nothing, kwargs...)
    A    = T(1)
    B    = T(3)
    L    = T(4)
    TMIN = T(0.25)
    h    = T(1) / T(NH)
    h2   = h / T(2)
    absBA = abs(B - A)
    c4absBA = T(4) * absBA   # = 8.0

    nv = 2 * (NH + 1)
    lvar = fill(T(-Inf), nv)
    uvar = fill(T(Inf), nv)

    # Fix X(0) = A, X(NH) = B
    lvar[1]    = A;  uvar[1]    = A
    lvar[NH+1] = B;  uvar[NH+1] = B

    # Start: X(K)=c4absBA*(K/NH)*(0.5*(K/NH)-TMIN)+A, U(K)=c4absBA*((K/NH)-TMIN)
    start = zeros(T, nv)
    for k in 0:NH
        t = T(k) / T(NH)
        start[k+1]       = c4absBA * t * (T(0.5) * t - TMIN) + A   # X(K)
        start[NH+1+k+1]  = c4absBA * (t - TMIN)                     # U(K)
    end

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, nv; lvar = lvar, uvar = uvar, start = start)

    # Objective: trapezoidal integration of X*sqrt(1+U^2)
    ExaModels.@add_obj(c, h2 * x[1] * sqrt(T(1) + x[NH+2]^2))                      # I=0
    ExaModels.@add_obj(c, h * x[i+1] * sqrt(T(1) + x[NH+1+i+1]^2) for i in 1:NH-1) # I=1..NH-1
    ExaModels.@add_obj(c, h2 * x[NH+1] * sqrt(T(1) + x[2*NH+2]^2))                 # I=NH

    # D(I) constraints: X(I)-X(I+1)+H/2*(U(I)+U(I+1))=0 for I=0..NH-1
    ExaModels.@add_con(c, x[i+1] - x[i+2] + h2 * (x[NH+1+i+1] + x[NH+1+i+2]) for i in 0:NH-1)

    # LENGTH constraint: trapezoidal integration of sqrt(1+U^2) - L = 0
    ExaModels.@add_con(c,
        h2 * sqrt(T(1) + x[NH+2]^2) +
        sum(h * sqrt(T(1) + x[NH+1+i+1]^2) for i in 1:NH-1) +
        h2 * sqrt(T(1) + x[2*NH+2]^2) - L
    )

    return ExaModels.ExaModel(c; kwargs...)
end
