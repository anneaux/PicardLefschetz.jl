using Test
using PicardLefschetz
using PicardLefschetz.Types
using PicardLefschetz.Thimble
using PicardLefschetz.Saddle

@testset "Thimbles Core Methods" begin

    # 1D Setup
    phase_1d(x::Vector) = t -> x[1] * t + x[2] * t^2 + x[3] * t^3 + t^5
    phase_drv_1d(x::Vector) = t -> x[1] + 2 * x[2] * t + 3 * x[3] * t^2 + 5 * t^4
    phase_hess_1d(x::Vector) = t -> 2 * x[2] + 6 * x[3] * t + 20 * t^3
    params_1d = [2., 3., 4.]
    domain_1d = [RealDomain(-5.0, 5.0)]
    complex_domain_1d = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]
    saddle_params_1d = Dict(
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
        "init_subdivision" => 10
    )

    # 2D Setup
    phase_2d(x::Vector) = t -> x[1] * t[1]^2 + x[2] * t[2]^2 + t[1] * t[2]
    phase_drv_2d(x::Vector) = t -> [2 * x[1] * t[1] + t[2], 2 * x[2] * t[2] + t[1]]
    phase_hess_2d(x::Vector) = t -> [2*x[1] 1; 1 2*x[2]]
    params_2d = [1.0, 1.0]
    domain_2d = [RealDomain(-5.0, 5.0), RealDomain(-5.0, 5.0)]
    complex_domain_2d = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im), ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]
    saddle_params_2d = Dict(
        "point_count" => 10,
        "accuracy" => 5,
        "grid_resolution" => 10,
        "flow_step_factor" => 0.1,
        "initial_necklace_size" => 10,
        "max_iterations" => 10,
        "init_perturbation_radius" => 0.01,
        "subdivision_threshold" => 0.1,
        "gradient_normalisation_threshold" => 1e-4,
        "height_threshold" => 0.01,
        "initial_point_count" => 10,
        "flow_steps" => 10,
        "max_simplices" => 100,
        "simplex_tolerance" => 10,
        "init_point_count" => 10
    )

    @testset "1D get_thimble!" begin
        saddles = find_saddles(phase_drv_1d(params_1d), complex_domain_1d, saddle_params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_thimble!(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), saddle, saddle_params_1d, mesh_type="none")
            @test saddle.thimble !== nothing
            @test saddle.thimble isa Tuple{Vector{<:FlowPoint},Vector{<:Simplex}}
        end
    end

    @testset "1D get_thimbles" begin
        # Passing real domains for 1D as per Thimbles method signature
        thimbles = get_thimbles(phase_1d(params_1d), phase_drv_1d(params_1d), saddle_params_1d, domain_1d, false, mesh_type="none")
        @test thimbles !== nothing
        @test thimbles isa Tuple{Vector{<:FlowPoint},Vector{<:Simplex}}
    end

    @testset "1D get_thimble_boundary!" begin
        saddles = find_saddles(phase_drv_1d(params_1d), complex_domain_1d, saddle_params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_thimble_boundary!(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), saddle, saddle_params_1d, mesh_type="none")
            @test saddle.thimble_boundary !== nothing
            @test saddle.thimble_boundary isa Vector{<:AbstractVector}
        end
    end

    @testset "1D get_thimble_boundaries" begin
        saddles_with_boundaries = get_thimble_boundaries(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), complex_domain_1d, saddle_params_1d, false, mesh_type="none")
        @test saddles_with_boundaries isa Vector
    end

    @testset "2D get_thimble! (quad)" begin
        saddles = find_saddles(phase_drv_2d(params_2d), complex_domain_2d, saddle_params_2d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_thimble!(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), saddle, saddle_params_2d, mesh_type="quad")
            @test saddle.thimble !== nothing
            @test saddle.thimble isa Vector{<:Simplex}
        end
    end

    @testset "2D get_thimbles (quad)" begin
        thimbles = get_thimbles(phase_2d(params_2d), phase_drv_2d(params_2d), saddle_params_2d, domain_2d, false, mesh_type="quad")
        @test thimbles !== nothing
        @test thimbles isa Tuple{Vector{<:Simplex}, Vector{<:FlowPoint}, Vector{<:Simplex}}
    end

    @testset "2D get_thimble_boundary! (quad)" begin
        saddles = find_saddles(phase_drv_2d(params_2d), complex_domain_2d, saddle_params_2d)
        if !isempty(saddles)
            saddle = saddles[1]
            get_thimble_boundary!(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), saddle, saddle_params_2d, mesh_type="quad")
            @test saddle.thimble_boundary !== nothing
            @test saddle.thimble_boundary isa Vector{<:Simplex}
        end
    end

    @testset "2D get_thimble_boundaries (quad)" begin
        saddles_with_boundaries = get_thimble_boundaries(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), complex_domain_2d, saddle_params_2d, false, mesh_type="quad")
        @test saddles_with_boundaries isa Vector
    end
end
