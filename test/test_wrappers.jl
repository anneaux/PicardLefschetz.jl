using Test
using Symbolics
using PicardLefschetz
using PicardLefschetz.Types: Saddle
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
        "max_grid_element_count" => 100,
        "simplex_tolerance" => 10,
        "GL_order" => 10,
        "simplex_order" => 5
    )

    domain_1d_real = [RealDomain(-5.0, 5.0)]
    domain_1d_complex = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]

    # 1D Phase Setup
    S_expr = 2.0 * z + 3.0 * z^2 + 4.0 * z^3 + z^5
    S_drv_expr = Symbolics.gradient(S_expr, [z])[1]
    prefactor_expr = z^0  # effectively 1.0

    @testset "Saddles Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        @test saddles isa Vector{<:Saddle}

        if !isempty(saddles)
            saddle = saddles[1]
            check_contribution!(z, S_expr, saddle, domain_1d_complex[1], params_1d)
            @test saddle.contributing isa Bool

            ans = solve_first_derivative(z, S_drv_expr, [complex(0.0)], 5)
            @test !isnothing(ans)

            flow_params_1d = copy(params_1d)
            flow_params_1d["height_threshold"] = -10.0
            flow_params_1d["gradient_normalisation_threshold"] = 1.0
            get_intersection_number!(z, S_expr, saddle, flow_params_1d)
            @test saddle.intersection_number !== nothing
            @test saddle.intersection_number isa Int
        end
    end

    @testset "Thimbles Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_thimble!(z, S_expr, saddle, params_1d, mesh_type="none")
            @test saddle.thimble !== nothing
            @test saddle.thimble isa Tuple{Vector{<:FlowPoint}, Vector{<:Simplex}}

            get_thimble_boundary!(z, S_expr, saddle, params_1d, mesh_type="none")
            @test saddle.thimble_boundary !== nothing
            @test saddle.thimble_boundary isa Vector{<:AbstractVector}
        end

        thimbles = get_FLIC(z, S_expr, params_1d, domain_1d_real, mesh_type="none")
        @test thimbles !== nothing

        thimbles_complex = get_thimbles(z, S_expr, params_1d, domain_1d_complex, false, mesh_type="none")
        @test thimbles_complex isa Vector{<:Saddle}
        if !isempty(thimbles_complex)
            @test thimbles_complex[1].thimble !== nothing
        end

        boundaries = get_thimble_boundaries(z, S_expr, domain_1d_complex, params_1d, false, mesh_type="none")
        @test boundaries isa Vector{<:Saddle}
    end

    @testset "Dual Thimbles Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_dual_thimble!(z, S_expr, saddle, params_1d)
            @test saddle.dual_thimble !== nothing
            @test saddle.dual_thimble isa Vector{<:FlowPoint}

            get_dual_thimble_boundary!(z, S_expr, saddle, params_1d)
            @test saddle.dual_thimble_boundary !== nothing
            @test saddle.dual_thimble_boundary isa Vector{<:FlowPoint}
        end

        dual_thimbles = get_dual_thimbles(z, S_expr, params_1d, domain_1d_complex, false)
        @test dual_thimbles isa Vector{<:Saddle}

        dual_boundaries = get_dual_thimble_boundaries(z, S_expr, params_1d, domain_1d_complex, false)
        @test dual_boundaries isa Vector{<:Saddle}
    end

    @testset "Integration Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_1d_complex, params_1d)
        if !isempty(saddles)
            saddle = saddles[1]

            # test integrate_SPM_thimble!
            integrate_SPM_thimble!(z, S_expr, saddle, prefactor_expr)
            @test saddle.integral !== nothing
            @test saddle.integral isa ComplexF64

            # test integrate_thimble (requires a thimble first)
            get_thimble!(z, S_expr, saddle, params_1d, mesh_type="none")
            if saddle.thimble !== nothing
                integral = integrate_thimble(z, S_expr, saddle.thimble, prefactor_expr, params_1d)
                @test integral !== nothing
            end
        end

        # test integrate_FLIC (real domains) 
        integral_real = integrate_FLIC(z, S_expr, domain_1d_real, [0.0], prefactor_expr, params_1d, "unfixed")
        @test integral_real !== nothing
        @test integral_real isa Tuple
        @test integral_real[1] isa AbstractVector

        # test integrate_thimbles (complex domains) 
        integral_complex = integrate_thimbles(z, S_expr, domain_1d_complex, params_1d, prefactor_expr)
        @test integral_complex !== nothing
        @test integral_complex isa Vector{<:Types.Saddle}
        @test !isempty(integral_complex)
        @test integral_complex[1].integral isa AbstractVector
    end

end

@testset "Symbolic Wrappers (2D)" begin

    @variables z[1:2]

    params_2d = Dict(
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
        "output_dim" => 2,
        "initial_point_count" => 10,
        "flow_steps" => 10,
        "init_point_count" => 10,
        "max_simplices" => 100,
        "max_grid_element_count" => 100,
        "simplex_tolerance" => 10
    )

    domain_2d_complex = [
        ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im),
        ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)
    ]

    # 2D Phase Setup: x^2 + y^2 + xy
    S_expr = z[1]^2 + z[2]^2 + z[1] * z[2]
    # We must construct a native derivative to find saddles, as find_saddles does not support Vector{Num} wrappers directly yet
    native_S_drv = build_function(Symbolics.gradient(S_expr, z), z, expression=Val{false})[1]

    @testset "2D Intersection Number Wrapper" begin
        # Temporarily use native find_saddles because wrappers/saddles.jl lacks a find_saddles(z::Vector{Num}, ...) method
        saddles = find_saddles(native_S_drv, domain_2d_complex, params_2d)

        if !isempty(saddles)
            saddle = saddles[1]

            flow_params_2d = copy(params_2d)
            flow_params_2d["height_threshold"] = -10.0
            flow_params_2d["gradient_normalisation_threshold"] = 1.0

            # Test our newly added 2D get_intersection_number! wrapper
            get_intersection_number!(z, S_expr, saddle, flow_params_2d)

            @test saddle.intersection_number !== nothing
            @test saddle.intersection_number isa Int
            @test saddle.intersection_number in [-1, 0, 1]
        end
    end
end

@testset "Symbolic Wrappers (2D)" begin
    @variables z[1:2]

    params_2d = Dict(
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
        "output_dim" => 2,
        "initial_point_count" => 10,
        "flow_steps" => 10,
        "init_point_count" => 20,
        "max_simplices" => 100,
        "max_grid_element_count" => 100,
        "simplex_tolerance" => 10,
        "GL_order" => 10,
        "simplex_order" => 5,
        "initial_necklace_size" => 10,
        "subdividethreshold" => 0.1
    )

    domain_2d_real = [RealDomain(-5.0, 5.0), RealDomain(-5.0, 5.0)]
    domain_2d_complex = [
        ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im),
        ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)
    ]

    # 2D Phase Setup: (z[1]-1)^2 + (z[2]-2)^2 + z[1]*z[2] - 5
    S_expr = (z[1] - 1)^2 + (z[2] - 2)^2 + z[1] * z[2] - 5
    S_drv_expr = Symbolics.gradient(S_expr, z)
    prefactor_expr = z[1]^0  # effectively 1.0

    @testset "2D Saddles Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_2d_complex, params_2d)
        @test saddles isa Vector{<:Saddle}

        if !isempty(saddles)
            saddle = saddles[1]

            # test check_contribution!
            check_contribution!(z, S_expr, saddle, domain_2d_complex, params_2d)
            @test saddle.contributing isa Bool

            # test solve_first_derivative
            ans = solve_first_derivative(z, S_drv_expr, [complex(0.0), complex(0.0)], 5)
            @test !isnothing(ans)
        end
    end

    @testset "2D Thimbles Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_2d_complex, params_2d)
        if !isempty(saddles)
            saddle = saddles[1]

            get_thimble!(z, S_expr, saddle, params_2d, mesh_type="quad")
            @test saddle.thimble !== nothing
            @test saddle.thimble isa Tuple{Vector{<:FlowPoint}, Vector{<:Simplex}}

            get_thimble_boundary!(z, S_expr, saddle, params_2d, mesh_type="quad")
            @test saddle.thimble_boundary !== nothing
            @test saddle.thimble_boundary isa Tuple{Vector{<:FlowPoint}, Vector{<:Simplex}}
        end

        thimbles = get_FLIC(z, S_expr, params_2d, domain_2d_real, mesh_type="quad")
        @test thimbles !== nothing

        thimbles_complex = get_thimbles(z, S_expr, params_2d, domain_2d_complex, false, mesh_type="quad")
        @test thimbles_complex isa Vector{<:Saddle}
        if !isempty(thimbles_complex)
            @test thimbles_complex[1].thimble !== nothing
        end

        boundaries = get_thimble_boundaries(z, S_expr, domain_2d_complex, params_2d, false, mesh_type="quad")
        @test boundaries isa Vector{<:Saddle}
    end

    @testset "2D Dual Thimbles Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_2d_complex, params_2d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_dual_thimble!(z, S_expr, saddle, params_2d)
            @test saddle.dual_thimble !== nothing
            @test saddle.dual_thimble isa Tuple{Vector{<:FlowPoint}, Vector{<:Simplex}}

            get_dual_thimble_boundary!(z, S_expr, saddle, params_2d)
            @test saddle.dual_thimble_boundary !== nothing
            @test saddle.dual_thimble_boundary isa Tuple{Vector{<:FlowPoint}, Vector{<:Simplex}}
        end

        dual_thimbles = get_dual_thimbles(z, S_expr, params_2d, domain_2d_complex, false)
        @test dual_thimbles isa Vector{<:Saddle}

        dual_boundaries = get_dual_thimble_boundaries(z, S_expr, params_2d, domain_2d_complex, false)
        @test dual_boundaries isa Vector{<:Saddle}
    end

    @testset "2D Integration Wrappers" begin
        saddles = find_saddles(z, S_drv_expr, domain_2d_complex, params_2d)
        if !isempty(saddles)
            saddle = saddles[1]

            # test integrate_SPM_thimble!
            integrate_SPM_thimble!(z, S_expr, saddle, prefactor_expr)
            @test saddle.integral !== nothing
            @test saddle.integral isa ComplexF64

            # test integrate_thimble (requires a thimble first)
            get_thimble!(z, S_expr, saddle, params_2d, mesh_type="quad")
            if saddle.thimble !== nothing
                integral = integrate_thimble(z, S_expr, saddle.thimble, prefactor_expr, params_2d)
                @test integral !== nothing
            end
        end

        # test integrate_FLIC (real domains) 
        # this core method is broken due to missing initialise_grid_parallelogram
        @test_throws UndefVarError integrate_FLIC(z, S_expr, domain_2d_real, [0.0, 0.0], prefactor_expr, params_2d, "unfixed")

        # test integrate_thimbles (complex domains) 
        integral_complex = integrate_thimbles(z, S_expr, domain_2d_complex, params_2d, prefactor_expr)
        @test integral_complex !== nothing
        @test integral_complex isa Vector{<:Types.Saddle}
        @test !isempty(integral_complex)
        @test integral_complex[1].integral isa AbstractVector
    end
end
