# test/runtests.jl — ExaCUTEst against CUTEst.jl.
#
# Two layers. The API layer exercises collection/selection/parameter handling
# with no reference needed. The agreement layer compares a decoded model
# against a CUTEstModel of the same problem: values, gradients, constraints,
# Jacobian and Lagrangian Hessian, permuted by SIF NAME, because CUTEst
# reorders both variables and rows while a decoded model keeps SIF
# declaration order.
#
# The full-collection validation lives in ExaModelsSIF (test/valsweep.jl,
# 1548/1548); this suite is the wrapper's own contract.
# NOTE: ExaCUTEst and CUTEst export the same names on purpose (that is what
# makes porting a rename), so with BOTH loaded Julia leaves those names
# unbound and every call must be qualified. Only code comparing the two ever
# loads both; `import` rather than `using` keeps that explicit here.
import ExaCUTEst
using ExaCUTEst: ExaCUTEstModel, row_linearity, sif_variable_names,
    sif_constraint_names, classification, is_truly_unconstrained,
    external_fortran_problems, sif_problem_path, native_problems
using Test, LinearAlgebra, SparseArrays, Random
using NLPModels
using CUTEst
using FileWatching.Pidfile: mkpidlock

# CUTEst decodes and compiles every problem into one shared directory, so
# concurrent CUTEstModel builds on one machine collide. Every construction
# holds this lock.
const CUTEST_LOCK = "/tmp/cutest-decode.pid"

# HS68 and HS69 are in this list deliberately: they are external-Fortran
# problems served by hand-written native models, so agreement with CUTEst is
# the ONLY thing standing between them and a silently wrong model. For the
# decoded problems the full-collection sweep upstream already covers that.
const AGREE = ["ROSENBR", "HS14", "HS35", "HS45", "DECONVC", "BROWNAL",
               "GENHS28", "DEGDIAG", "AVGASB", "LUKVLE14", "HS68", "HS69"]

relerr(a, b) = begin
    m = isfinite.(a)
    any(m) ? maximum(abs.(a[m] .- b[m]) ./ (1 .+ max.(abs.(a[m]), abs.(b[m])))) : 0.0
end

# relerr over the union of two sparse matrices' patterns. Densifying is not an
# option: DEGDIAG has 100001 variables, so its dense Hessian is 80 GB.
function sprelerr(A, B)
    I, J, _ = findnz(abs.(A) .+ abs.(B))
    return relerr([A[i, j] for (i, j) in zip(I, J)], [B[i, j] for (i, j) in zip(I, J)])
end

# SIF names as the Fortran reference reports them. The buffers must be
# Cchar: a UInt8 matrix finds no method.
_decolumn(buf, k) = String(strip(join(Char(UInt8(c)) for c in buf[:, k])))

function refnames(ref, which)
    n, m = ref.meta.nvar, ref.meta.ncon
    status = zeros(Cint, 1)
    vname = Matrix{Cchar}(undef, 10, n)
    if which === :var
        CUTEst.varnames(Float64, ref.libsif, status, Cint[n], vname)
        return [_decolumn(vname, j) for j in 1:n]
    end
    pname = Vector{Cchar}(undef, 10)
    cname = Matrix{Cchar}(undef, 10, m)
    CUTEst.cnames(Float64, ref.libsif, status, Cint[n], Cint[m], pname, vname, cname)
    return [_decolumn(cname, j) for j in 1:m]
end

@testset "ExaCUTEst" begin
    @testset "collection" begin
        probs = ExaCUTEst.list_sif_problems()
        @test length(probs) > 1500
        @test "ROSENBR" in probs
        @test isfile(sif_problem_path("ROSENBR"))
        @test_throws ErrorException sif_problem_path("NOT_A_PROBLEM_XYZ")
        @test first(ExaCUTEst.generator_sif_problems(; filter = n -> n == "ROSENBR")) == "ROSENBR"
    end

    @testset "collection switching" begin
        # each set must resolve to its OWN directory. Hardcoding the
        # artifacts' inner directory names (taken from an older CUTEst.jl)
        # gave a path that does not exist, and mastsif() then silently served
        # the default 1548-problem collection -- a wrong answer whose only
        # symptom was a plausible-looking count.
        main = ExaCUTEst.set_mastsif("sifcollection")
        nmain = length(ExaCUTEst.list_sif_problems())
        mm = ExaCUTEst.set_mastsif("maros-meszaros")
        nmm = length(ExaCUTEst.list_sif_problems())
        @test isdir(main) && isdir(mm)
        @test main != mm
        @test nmain > 1500
        @test 100 < nmm < 500          # 138 today; not the main collection
        @test nmm != nmain
        @test "AUG2D" in ExaCUTEst.list_sif_problems()      # a Maros-Meszaros QP
        @test_throws ErrorException ExaCUTEst.set_mastsif("not-a-set")

        # a MASTSIF that is set but wrong must error, not fall back
        saved = ENV["MASTSIF"]
        ENV["MASTSIF"] = joinpath(saved, "definitely_not_here")
        @test_throws ErrorException ExaCUTEst.mastsif()
        ENV["MASTSIF"] = main
        @test ExaCUTEst.mastsif() == main
    end

    @testset "parameter arguments" begin
        # the three sifdecoder spellings must agree, and with the params kwarg
        a = ExaCUTEstModel("READING4", "-param", "N=100")
        b = ExaCUTEstModel("READING4", "N=100")
        c = ExaCUTEstModel("READING4", "-param", "N", "100")
        d = ExaCUTEstModel("READING4"; params = Dict("N" => 100))
        @test a.meta.nvar == b.meta.nvar == c.meta.nvar == d.meta.nvar == 101
        # and must actually change the problem
        @test ExaCUTEstModel("READING4").meta.nvar != a.meta.nvar
        @test_throws ErrorException ExaCUTEstModel("READING4", "-param", "N")
    end

    @testset "selection" begin
        every = ExaCUTEst.select_sif_problems()
        @test length(every) > 1000
        small = ExaCUTEst.select_sif_problems(; max_var = 10)
        @test "ROSENBR" in small
        @test issubset(Set(small), Set(every))
        @test all(p -> classification(p)["variables"]["number"] <= 10, small)
        sos = ExaCUTEst.select_sif_problems(; objtype = :sum_of_squares)
        @test all(p -> classification(p)["objtype"] == "sum_of_squares", sos)
        @test sos == ExaCUTEst.select_sif_problems(; objtype = 5)
        @test isempty(intersect(sos, ExaCUTEst.select_sif_problems(; objtype = :linear)))
        @test is_truly_unconstrained("ROSENBR")
        @test !is_truly_unconstrained("HS14")
    end

    @testset "row linearity" begin
        # meta cannot answer this (ExaModels reports every row nonlinear),
        # which is the whole reason row_linearity exists
        @test row_linearity("HS14") == (nlin = 1, nnln = 1)
        @test ExaCUTEstModel("HS14").meta.nlin == 0
        @test row_linearity("HS35").nlin == 1
        @test row_linearity("ROSENBR") == (nlin = 0, nnln = 0)
    end

    @testset "external Fortran" begin
        ext = external_fortran_problems()
        @test length(ext) == 34
        @test "AKIVA" in ext
        # every external-Fortran problem has a native model, and only those do
        @test native_problems() == sort(ext)
        nlp = ExaCUTEstModel("AKIVA")
        @test isfinite(obj(nlp, nlp.meta.x0))
        @test eltype(ExaCUTEstModel("AKIVA"; T = Float32).meta.x0) === Float32
        # a native model is independent of any other built from the same name
        nlp2 = ExaCUTEstModel("AKIVA")
        @test obj(nlp2, nlp2.meta.x0) == obj(nlp, nlp.meta.x0)
        # with parameter overrides the native model does not apply, and the
        # decoder refuses the external Fortran rather than guessing
        @test_throws ExaCUTEst.ExaModelsSIF.ExternalFortranError ExaCUTEstModel(
            "AKIVA", "-param", "N=3")
    end

    @testset "agreement with CUTEst.jl: $name" for name in AGREE
        ref = mkpidlock(() -> CUTEstModel(name), CUTEST_LOCK; wait = true)
        try
            nlp = ExaCUTEstModel(name)
            @test nlp.meta.nvar == ref.meta.nvar
            @test nlp.meta.ncon == ref.meta.ncon

            # Models are in SIF declaration order, native ones included, so
            # they are permuted by name onto CUTEst's ordering.
            vp = indexin(refnames(ref, :var), sif_variable_names(name))
            perm = ref.meta.ncon > 0 ?
                indexin(refnames(ref, :con), sif_constraint_names(name)) : Int[]
            @test !any(isnothing, vp)
            @test !any(isnothing, perm)

            Random.seed!(7)
            for x in (copy(ref.meta.x0),
                      clamp.(ref.meta.x0 .+ 0.01 .* randn(ref.meta.nvar),
                             ref.meta.lvar, ref.meta.uvar))
                xm = zeros(length(x)); xm[vp] = x
                @test obj(nlp, xm) ≈ obj(ref, x) rtol = 1e-8
                @test relerr(grad(ref, x), grad(nlp, xm)[vp]) < 1e-8
                if ref.meta.ncon > 0
                    @test relerr(cons(ref, x), cons(nlp, xm)[perm]) < 1e-8
                    Ja = sparse(jac_structure(ref)..., jac_coord(ref, x),
                                ref.meta.ncon, ref.meta.nvar)
                    Jb = sparse(jac_structure(nlp)..., jac_coord(nlp, xm),
                                nlp.meta.ncon, nlp.meta.nvar)[perm, vp]
                    @test sprelerr(Ja, Jb) < 1e-8
                end
                y = ref.meta.ncon > 0 ? randn(ref.meta.ncon) : Float64[]
                ym = zeros(length(y)); ym[perm] = y
                Ha = sparse(hess_structure(ref)..., hess_coord(ref, x, y),
                            ref.meta.nvar, ref.meta.nvar)
                Hb = sparse(hess_structure(nlp)..., hess_coord(nlp, xm, ym),
                            nlp.meta.nvar, nlp.meta.nvar)
                Ha = Ha + transpose(Ha)
                Hb = (Hb + transpose(Hb))[vp, vp]
                @test sprelerr(Ha, Hb) < 1e-8
            end
        finally
            finalize(ref)
        end
    end
end
