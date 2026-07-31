using PicardLefschetz
using Test
using LinearAlgebra


# We need to access some unexported internals for thorough testing
import PicardLefschetz: get_preset, unpack_params, get_hessian_eigenvectors!

@testset "Heuristics Tests" begin

    @testset "Presets" begin
        @test length(get_preset(:fast)) == 6
        @test length(get_preset(:accurate)) == 6
        @test length(get_preset(:thimble)) == 6
        @test_throws ArgumentError get_preset(:invalid_preset)
    end

    @testset "Unpack Params" begin
        p5 = [1.0, 2.0, 3.0, 4.0, 50.4]
        α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = unpack_params(p5)
        @test α_init == 1.0
        @test α_subdiv == 2.0
        @test α_grad_scale == 3.0
        @test α_grad == 4.0
        @test Nflow == 50
        @test h_threshold == -300.0

        p6 = [1.0, 2.0, 3.0, 4.0, 50.6, -500.0]
        α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = unpack_params(p6)
        @test Nflow == 51
        @test h_threshold == -500.0

        @test_throws ArgumentError unpack_params([1.0, 2.0])
    end

    @testset "Ball Radius" begin
        # S(z) = z^2. At ξ = 0, S(ξ) = 0.
        # We look for |S(r*d) - S(0)|^2 - Cω^2 = 0
        # |r^2|^2 - 1 = 0 => r = 1.0
        S_quad(z) = z^2
        r = ballradius(S_quad, 0.0 + 0.0im, 1.0, 1.0 + 0.0im; r_max=10.0)
        @test r ≈ 1.0 atol=1e-3

        # Test failure when r_max is too small
        @test_throws ErrorException ballradius(S_quad, 0.0 + 0.0im, 1.0, 1.0 + 0.0im; r_max=0.5)
    end

    @testset "Hessian Eigenvectors" begin
        # S(z) = z^2 = x^2 - y^2 + 2ixy
        # Im(S) = 2xy
        # Hessian of Im(S) = [0 2; 2 0]
        # Eigenvalues: 2, -2
        # Eigenvectors: [1, 1], [1, -1] (unnormalized from eigendecomp)
        S_quad(z) = z^2
        dirs_descent = ComplexF64[]
        get_hessian_eigenvectors!(dirs_descent, 0.0 + 0.0im, S_quad, :descent)
        
        # We expect 2 directions for the positive eigenvalue
        @test length(dirs_descent) == 2
        
        dirs_ascent = ComplexF64[]
        get_hessian_eigenvectors!(dirs_ascent, 0.0 + 0.0im, S_quad, :ascent)
        
        # We expect 2 directions for the negative eigenvalue
        @test length(dirs_ascent) == 2
    end

    @testset "Directional r_osc" begin
        S_quad(z) = z^2
        pairs = get_directional_r_osc(S_quad, 0.0 + 0.0im, 1.0; Cball=1.0, r_max=10.0)
        # Should return 4 directions (2 descent + 2 ascent)
        @test length(pairs) == 4
        for (d_norm, r) in pairs
            @test r ≈ 1.0 atol=1e-3
            @test norm(d_norm) ≈ 1.0 atol=1e-6
        end
    end

    @testset "get_pl_heuristics_1d and get_pl_heuristics_2d" begin
        S_quad(z) = z^2
        drv_quad(z) = 2z
        
        # Test 1D version
        params_1d = get_pl_heuristics_1d(S_quad, drv_quad, 0.0 + 0.0im, 1.0, :fast)
        @test haskey(params_1d, :Δinit)
        @test haskey(params_1d, :r_osc)
        @test haskey(params_1d, :Nflow)
        @test params_1d.Nflow == 50

        # S_2d(z1, z2) = z1^2 + z2^2
        # Im(S) = 2*x1*y1 + 2*x2*y2
        # Hessian w.r.t (x1, y1, x2, y2) is a 4x4 real matrix
        S_2d(z1, z2) = z1^2 + z2^2
        f_grad(z1, z2) = [2z1, 2z2]
        f_hessian(z1, z2) = [0.0 2.0 0.0 0.0; 2.0 0.0 0.0 0.0; 0.0 0.0 0.0 2.0; 0.0 0.0 2.0 0.0]
        
        params_2d = get_pl_heuristics_2d(S_2d, f_grad, f_hessian, [0.0+0.0im, 0.0+0.0im], 1.0, :accurate)
        @test haskey(params_2d, :eigvecfactorinit)
        @test haskey(params_2d, :r_osc)
        @test haskey(params_2d, :Nflow)
        # Test resolve_heuristics standalone
        res = resolve_heuristics(:fast, Nflow=999)
        @test res.Nflow == 999
        @test res.α_init == 0.36964

        # Test feeding pre-computed params into get_thimble via params keyword
        pts, sims = get_thimble(S_quad, drv_quad, -2.0, 2.0; params=params_1d)
        @test !isempty(pts)
        @test !isempty(sims)
    end
end
