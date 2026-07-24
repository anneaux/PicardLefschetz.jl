using Test
using Symbolics
using PicardLefschetz
using PicardLefschetz.Types
using PicardLefschetz.Saddle
using PicardLefschetz.Thimble
using PicardLefschetz.DualThimble
using PicardLefschetz.Integration

@testset "Symbolic Wrappers (1D)" begin
    @variables z

    # Basic 1D parameters, matching test_thimbles.jl setup
    params_1d = Dict(
        "point_count" => 10,
        "accuracy" => 5,
        "grid_resolution" => 10,
        "init_perturbation_radius" => 0.01,
        "max_iterations" => 10,
        "flow_step_factor" => 0.1,
        "subdivision_threshold" => 0.1,
        "gradient_normalisation_threshold" => 1e-4,
        "height_threshold" => 0.01,
        "iterations" => 10,
        "init_subdivision" => 10,
        "grid_spacing" => 0.1,
        "flow_rate_scaling" => 0.1,
        "gradient_normalisation_factor" => 1e-4,
        "max_flow_steps" => 10,
        "integral_accuracy" => 1e-3,
        "integral_relative_error" => 1e-3,
        "verbose" => false,
        "output_dim" => 1,
        "initial_point_count" => 10,
        "flow_steps" => 10,
        "init_point_count" => 10,
        "max_simplices" => 100,
        "simplex_tolerance" => 10
    )

    domain_1d_real = [RealDomain(-5.0, 5.0)]
    domain_1d_complex = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]

    # 1D Phase Setup
    S_expr = 2.0 * z + 3.0 * z^2 + 4.0 * z^3 + z^5
    S_drv_expr = Symbolics.gradient(S_expr, [z])[1]
    prefactor_expr = z^0  # effectively 1.0

    @testset "Saddles Wrappers" begin
        saddles = find_numerical_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        @test saddles isa Vector{Saddle}

        if !isempty(saddles)
            saddle = saddles[1]
            check_contribution!(z, S_expr, saddle, domain_1d_complex[1], params_1d)
            @test saddle.contributing isa Bool

            ans = find_analytic_saddles(z, S_drv_expr, [complex(0.0)], 5)
            @test !isnothing(ans)
        end
    end

    @testset "Thimbles Wrappers" begin
        saddles = find_numerical_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_thimble!(z, S_expr, saddle, params_1d, mesh_type="none")
            @test saddle.thimble !== nothing

            get_thimble_boundary!(z, S_expr, saddle, params_1d, mesh_type="none")
            @test saddle.thimble_boundary !== nothing
        end

        thimbles = get_thimbles(z, S_expr, params_1d, domain_1d_real, false, mesh_type="none")
        @test thimbles !== nothing

        boundaries = get_thimble_boundaries(z, S_expr, domain_1d_complex, params_1d, false, mesh_type="none")
        @test boundaries isa Vector{Saddle}
    end

    @testset "Dual Thimbles Wrappers" begin
        saddles = find_numerical_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_dual_thimble!(z, S_expr, saddle, params_1d)
            @test saddle.dual_thimble !== nothing

            get_dual_thimble_boundary!(z, S_expr, saddle, params_1d)
            @test saddle.dual_thimble_boundary !== nothing
        end

        dual_thimbles = get_dual_thimbles(z, S_expr, params_1d, domain_1d_complex, false)
        @test dual_thimbles isa Vector{Saddle}

        dual_boundaries = get_dual_thimble_boundaries(z, S_expr, params_1d, domain_1d_complex, false)
        @test dual_boundaries isa Vector{Saddle}
    end

    @testset "Integration Wrappers" begin
        saddles = find_numerical_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        if !isempty(saddles)
            saddle = saddles[1]

            # test integrate_thimble!
            integrate_thimble!(z, S_expr, saddle, prefactor_expr)
            @test saddle.integral !== nothing

            # test integrate_thimble (requires a boundary first)
            get_thimble_boundary!(z, S_expr, saddle, params_1d, mesh_type="none")
            if saddle.thimble_boundary !== nothing
                integral = integrate_thimble(z, S_expr, saddle.thimble_boundary, prefactor_expr, params_1d)
                @test integral !== nothing
            end
        end

        # test integrate_thimbles (real domains) 
        integral_real = integrate_thimbles(z, S_expr, domain_1d_real, [0.0], prefactor_expr, params_1d, "unfixed")
        @test integral_real !== nothing

        # test integrate_thimbles (complex domains) 
        integral_complex = integrate_thimbles(z, S_expr, domain_1d_complex, params_1d, prefactor_expr)
        @test integral_complex !== nothing
    end
end
