using Test, ExaCUTEst, CUTEst, NLPModels

const atol = 1e-10

function compare_with_cutest(m, name; x = NLPModels.get_x0(m))
    ref = CUTEstModel(name)
    try
        xr = ref.meta.x0

        @test m.meta.nvar == ref.meta.nvar
        @test m.meta.ncon == ref.meta.ncon
        @test NLPModels.get_x0(m) ≈ xr atol = atol
        @test m.meta.lvar ≈ ref.meta.lvar atol = atol
        @test m.meta.uvar ≈ ref.meta.uvar atol = atol
        @test NLPModels.obj(m, xr) ≈ NLPModels.obj(ref, xr) atol = atol
        @test NLPModels.grad(m, xr) ≈ NLPModels.grad(ref, xr) atol = atol
        if m.meta.ncon > 0
            @test m.meta.lcon ≈ ref.meta.lcon atol = atol
            @test m.meta.ucon ≈ ref.meta.ucon atol = atol
            @test NLPModels.cons(m, xr) ≈ NLPModels.cons(ref, xr) atol = atol
        end
    finally
        finalize(ref)
    end
end

@testset "ExaCUTEst" begin
    @testset "ROSENBR" begin
        compare_with_cutest(rosenbr_model(), "ROSENBR")
    end

    @testset "HS5" begin
        compare_with_cutest(hs5_model(), "HS5")
    end

    @testset "TAME" begin
        compare_with_cutest(tame_model(), "TAME")
    end

    @testset "ALSOTAME" begin
        compare_with_cutest(alsotame_model(), "ALSOTAME")
    end

    @testset "DEGDIAG" begin
        m = degdiag_model()
        ref = CUTEstModel("DEGDIAG")
        try
            xr = ref.meta.x0
            @test m.meta.nvar == ref.meta.nvar
            @test m.meta.ncon == ref.meta.ncon
            @test NLPModels.get_x0(m) ≈ xr atol = atol
            # lvar uses rtol due to floating-point rounding in i/(N+1) vs Fortran arithmetic
            @test m.meta.lvar ≈ ref.meta.lvar rtol = 1e-10
            @test NLPModels.obj(m, xr) ≈ NLPModels.obj(ref, xr) atol = atol
        finally
            finalize(ref)
        end
    end
end
