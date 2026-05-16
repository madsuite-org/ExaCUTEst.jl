# SPMSQRT: Liu and Nocedal tridiagonal matrix square root problem (feasibility NLP)
# Find tridiagonal X such that X^T X = A (a given pentadiagonal matrix)
# M = 1667, NV = 3*M-2 = 4999, NC = 5*M-6 = 8329 equality constraints
# classification NQR2-AN-V-V
# Source: Buckley (1989), problem 151. SIF input: Ph. Toint, Dec 1989.
#
# Variables: X(i,j) for the tridiagonal matrix
#   Row 1: X(1,1) -> x[1], X(1,2) -> x[2]
#   Row i (2..M-1): X(i,i-1)->x[3i-3], X(i,i)->x[3i-2], X(i,i+1)->x[3i-1]
#   Row M: X(M,M-1)->x[3M-3], X(M,M)->x[3M-2]
#
# Target matrix A = B*B where B is a given tridiagonal matrix with entries
#   filled sequentially: k=1→B(1,1)=sin(1); k=2→B(1,2)=sin(4);
#   k=3i-3→B(i,i-1), k=3i-2→B(i,i), k=3i-1→B(i,i+1) for i=2..M-1;
#   k=3M-3→B(M,M-1), k=3M-2→B(M,M)
#
# Constraint E(i,j): (X*X)[i,j] = A[i,j]
# Nonzero only for |i-j|<=2 (pentadiagonal), each constraint uses PROD2 (VI*VJ) and SQ (V^2) elements
function spmsqrt_model(; T = Float64, backend = nothing, kwargs...)
    M  = 1667
    NV = 3 * M - 2

    # B matrix entries (tridiagonal, stored by row)
    # Filled sequentially: k=1→B(1,1); k=2→B(1,2); k=3i-3→B(i,i-1); k=3i-2→B(i,i); k=3i-1→B(i,i+1) for i=2..M-1
    # k=3M-3→B(M,M-1); k=3M-2→B(M,M)
    function bval(i, j)
        if i == 1
            j == 1 && return sin(1.0)
            j == 2 && return sin(4.0)
        elseif i == M
            j == M-1 && return sin(Float64(3*M-3)^2)
            j == M   && return sin(Float64(3*M-2)^2)
        else  # 2 <= i <= M-1
            j == i-1 && return sin(Float64(3*i-3)^2)
            j == i   && return sin(Float64(3*i-2)^2)
            j == i+1 && return sin(Float64(3*i-1)^2)
        end
        return 0.0
    end

    # A[p,q] = (B*B)[p,q] = sum_k B[p,k]*B[k,q]
    function aval(p, q)
        total = 0.0
        for k in max(1, p-1, q-1):min(M, p+1, q+1)
            total += bval(p, k) * bval(k, q)
        end
        return total
    end

    # Starting point: 0.2 * B
    x0 = zeros(T, NV)
    x0[1] = T(0.2) * T(bval(1, 1));  x0[2] = T(0.2) * T(bval(1, 2))
    for i in 2:M-1
        x0[3*i-3] = T(0.2) * T(bval(i, i-1))
        x0[3*i-2] = T(0.2) * T(bval(i, i))
        x0[3*i-1] = T(0.2) * T(bval(i, i+1))
    end
    x0[3*M-3] = T(0.2) * T(bval(M, M-1));  x0[3*M-2] = T(0.2) * T(bval(M, M))

    c = ExaModels.ExaCore(T; backend = backend, concrete = Val(true))
    ExaModels.@add_var(c, x, NV; lvar = T(-Inf), uvar = T(Inf), start = x0)

    # ------- Row 1 constraints -------
    # E(1,1): D(1)+G(1) = X(1,1)^2 + X(1,2)*X(2,1) = x[1]^2 + x[2]*x[3]
    ExaModels.@add_con(c, x[1]^2 + x[2]*x[3] - T(aval(1,1)))
    # E(1,2): H(1)+R(1) = X(1,1)*X(1,2) + X(1,2)*X(2,2) = x[1]*x[2] + x[2]*x[4]
    ExaModels.@add_con(c, x[1]*x[2] + x[2]*x[4] - T(aval(1,2)))
    # E(1,3): S(1) = X(1,2)*X(2,3) = x[2]*x[5]
    ExaModels.@add_con(c, x[2]*x[5] - T(aval(1,3)))

    # ------- Row 2 constraints -------
    # E(2,1): B(2)+C(2) = X(2,1)*X(1,1) + X(2,2)*X(2,1) = x[3]*x[1] + x[4]*x[3]
    ExaModels.@add_con(c, x[3]*x[1] + x[4]*x[3] - T(aval(2,1)))
    # E(2,2): F(2)+D(2)+G(2) = X(2,1)*X(1,2) + X(2,2)^2 + X(2,3)*X(3,2)
    #   = x[3]*x[2] + x[4]^2 + x[5]*x[6]
    ExaModels.@add_con(c, x[3]*x[2] + x[4]^2 + x[5]*x[6] - T(aval(2,2)))
    # E(2,3): H(2)+R(2) = X(2,2)*X(2,3) + X(2,3)*X(3,3) = x[4]*x[5] + x[5]*x[7]
    ExaModels.@add_con(c, x[4]*x[5] + x[5]*x[7] - T(aval(2,3)))
    # E(2,4): S(2) = X(2,3)*X(3,4) = x[5]*x[8]
    ExaModels.@add_con(c, x[5]*x[8] - T(aval(2,4)))

    # ------- General rows i=3..M-2 -------
    # E(i,i-2): A(i) = X(i,i-1)*X(i-1,i-2) = x[3i-3]*x[3i-6]
    let Aim2 = [(a = T(aval(i, i-2)), vi1 = Int32(3*i-3), vi2 = Int32(3*i-6)) for i in 3:M-2]
        ExaModels.@add_con(c, x[d.vi1]*x[d.vi2] - d.a for d in Aim2)
    end

    # E(i,i-1) for i=3..M-1
    let Aim1 = [(a = T(aval(i, i-1)), vi1 = Int32(3*i-3), vi2 = Int32(3*i-5), vi3 = Int32(3*i-2)) for i in 3:M-1]
        ExaModels.@add_con(c, x[d.vi1]*x[d.vi2] + x[d.vi3]*x[d.vi1] - d.a for d in Aim1)
    end

    # E(i,i) for i=3..M-2
    let Aii = [(a = T(aval(i, i)), vi1 = Int32(3*i-3), vi2 = Int32(3*i-4), vi3 = Int32(3*i-2), vi4 = Int32(3*i-1), vi5 = Int32(3*i)) for i in 3:M-2]
        ExaModels.@add_con(c, x[d.vi1]*x[d.vi2] + x[d.vi3]^2 + x[d.vi4]*x[d.vi5] - d.a for d in Aii)
    end

    # E(i,i+1) for i=3..M-2
    let Aip1 = [(a = T(aval(i, i+1)), vi1 = Int32(3*i-2), vi2 = Int32(3*i-1), vi3 = Int32(3*i+1)) for i in 3:M-2]
        ExaModels.@add_con(c, x[d.vi1]*x[d.vi2] + x[d.vi2]*x[d.vi3] - d.a for d in Aip1)
    end

    # E(i,i+2) for i=3..M-2: S(i) = X(i,i+1)*X(i+1,i+2) = x[3i-1]*x[3i+2]
    let Aip2 = [(a = T(aval(i, i+2)), vi1 = Int32(3*i-1), vi2 = Int32(3*i+2)) for i in 3:M-2]
        ExaModels.@add_con(c, x[d.vi1]*x[d.vi2] - d.a for d in Aip2)
    end

    # ------- Row M-1 boundary constraints -------
    # E(M-1,M-3): A(M-1) = X(M-1,M-2)*X(M-2,M-3) = x[3M-6]*x[3M-9]
    ExaModels.@add_con(c, x[3*M-6]*x[3*M-9] - T(aval(M-1, M-3)))
    # (E(M-1,M-2) already covered by Aim1 loop at i=M-1)
    # E(M-1,M-1): F(M-1)+D(M-1)+G(M-1)
    #   F(M-1) = X(M-1,M-2)*X(M-2,M-1) = x[3M-6]*x[3M-7]
    #   D(M-1) = X(M-1,M-1)^2 = x[3M-5]^2
    #   G(M-1) = X(M-1,M)*X(M,M-1) = x[3M-4]*x[3M-3]
    ExaModels.@add_con(c, x[3*M-6]*x[3*M-7] + x[3*M-5]^2 + x[3*M-4]*x[3*M-3] - T(aval(M-1, M-1)))
    # E(M-1,M): H(M-1)+R(M-1)
    #   = X(M-1,M-1)*X(M-1,M) + X(M-1,M)*X(M,M)
    #   = x[3M-5]*x[3M-4] + x[3M-4]*x[3M-2]
    ExaModels.@add_con(c, x[3*M-5]*x[3*M-4] + x[3*M-4]*x[3*M-2] - T(aval(M-1, M)))

    # ------- Row M constraints -------
    # E(M,M-2): A(M) = X(M,M-1)*X(M-1,M-2) = x[3M-3]*x[3M-6]
    ExaModels.@add_con(c, x[3*M-3]*x[3*M-6] - T(aval(M, M-2)))
    # E(M,M-1): B(M)+C(M) = X(M,M-1)*X(M-1,M-1) + X(M,M)*X(M,M-1)
    #   = x[3M-3]*x[3M-5] + x[3M-2]*x[3M-3]
    ExaModels.@add_con(c, x[3*M-3]*x[3*M-5] + x[3*M-2]*x[3*M-3] - T(aval(M, M-1)))
    # E(M,M): F(M)+D(M) = X(M,M-1)*X(M-1,M) + X(M,M)^2
    #   = x[3M-3]*x[3M-4] + x[3M-2]^2
    ExaModels.@add_con(c, x[3*M-3]*x[3*M-4] + x[3*M-2]^2 - T(aval(M, M)))

    return ExaModels.ExaModel(c; kwargs...)
end
