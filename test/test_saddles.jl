using Test
using PicardLefschetz
using PicardLefschetz.Types
using PicardLefschetz.Saddle

@testset "Saddles Core Methods" begin

    # 1D Setup
    phase_1d(x::Vector) = t -> x[1] * t + x[2] * t^2 + x[3] * t^3 + t^5
    phase_drv_1d(x::Vector) = t -> x[1] + 2 * x[2] * t + 3 * x[3] * t^2 + 5 * t^4
    phase_hess_1d(x::Vector) = t -> 2 * x[2] + 6 * x[3] * t + 20 * t^3
    params_1d = [2., 3., 4.]
    domain_1d = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]
    saddle_params_1d = Dict{String,Any}(
        "point_count" => 100,
        "accuracy" => 5,
        "grid_resolution" => 10
    )

    # 2D Setup
    phase_2d(x::Vector) = t -> x[1] * t[1]^2 + x[2] * t[2]^2 + t[1] * t[2]
    phase_drv_2d(x::Vector) = t -> [2 * x[1] * t[1] + t[2], 2 * x[2] * t[2] + t[1]]
    phase_hess_2d(x::Vector) = t -> [2*x[1] 1; 1 2*x[2]]
    params_2d = [1.0, 1.0]
    domain_2d = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im), ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]
    saddle_params_2d = Dict(
        "point_count" => 100,
        "accuracy" => 5,
        "grid_resolution" => 10,
        "flow_step_factor" => 0.1,
        "initial_necklace_size" => 10,
        "max_iterations" => 100,
        "init_pertubation_radius" => 0.01,
        "subdividethreshold" => 0.1
    )

    @testset "1D Numerical Saddles" begin
        saddles = find_saddles(phase_drv_1d(params_1d), domain_1d, saddle_params_1d)
        @test saddles isa Vector
        # We expect up to 4 saddles for a degree 4 derivative
        @test length(saddles) <= 4
    end

    @testset "1D Analytic Saddles" begin
        # Provide an initial guess close to origin
        saddle = solve_first_derivative(phase_drv_1d(params_1d), [0.0 + 0.0im], 5)
        @test saddle isa Vector
    end

    @testset "1D Contribution Check" begin
        saddles = find_saddles(phase_drv_1d(params_1d), domain_1d, saddle_params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            check_contribution!(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), saddle, domain_1d[1], saddle_params_1d)
            @test saddle.contributing isa Bool
        end
    end

    @testset "2D Numerical Saddles" begin
        saddles = find_saddles(phase_drv_2d(params_2d), domain_2d, saddle_params_2d)
        @test saddles isa Vector
    end

    @testset "2D Analytic Saddles" begin
        # Provide an initial guess close to origin
        saddle = solve_first_derivative(phase_drv_2d(params_2d), [0.1 + 0.1im, 0.1 - 0.1im], 5)
        @test saddle isa Vector
    end

    @testset "2D Contribution Check" begin
        saddles = find_saddles(phase_drv_2d(params_2d), domain_2d, saddle_params_2d)
        if !isempty(saddles)
            saddle = saddles[1]
            check_contribution!(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), saddle, domain_2d[1], saddle_params_2d)
            @test saddle.contributing isa Bool
        end
    end

    @testset "1D Intersection Number" begin
        saddles = find_saddles(phase_drv_1d(params_1d), domain_1d, saddle_params_1d)
        if !isempty(saddles)
            saddle = saddles[1]

            # get_intersection_number! requires flow parameters to calculate the thimbles
            flow_params_1d = copy(saddle_params_1d)
            flow_params_1d["init_perturbation_radius"] = 0.01
            flow_params_1d["max_iterations"] = 100
            flow_params_1d["flow_step_factor"] = 0.1
            flow_params_1d["subdivision_threshold"] = 0.1
            flow_params_1d["height_threshold"] = -10.0
            flow_params_1d["gradient_normalisation_threshold"] = 1.0

            get_intersection_number!(
                phase_1d(params_1d),
                phase_drv_1d(params_1d),
                phase_hess_1d(params_1d),
                saddle,
                flow_params_1d
            )

            # Verify the intersection number is computed and is one of the valid topological intersection values
            @test saddle.intersection_number !== nothing
            @test saddle.intersection_number in [-1, 0, 1]

            # Since determinant != 0 implies contributing == true
            @test saddle.contributing == (saddle.intersection_number != 0)
        end
    end

    @testset "2D Intersection Number" begin
        saddles = find_saddles(phase_drv_2d(params_2d), domain_2d, saddle_params_2d)
        if !isempty(saddles)
            saddle = saddles[1]

            # get_intersection_number! requires flow parameters to calculate the thimbles
            flow_params_2d = copy(saddle_params_2d)
            flow_params_2d["init_perturbation_radius"] = 0.01
            flow_params_2d["max_iterations"] = 100
            flow_params_2d["flow_step_factor"] = 0.1
            flow_params_2d["subdivision_threshold"] = 0.1
            flow_params_2d["height_threshold"] = -10.0
            flow_params_2d["gradient_normalisation_threshold"] = 1.0

            get_intersection_number!(
                phase_2d(params_2d),
                phase_drv_2d(params_2d),
                saddle,
                flow_params_2d
            )

            # Verify the intersection number is computed and is one of the valid topological intersection values
            @test saddle.intersection_number !== nothing
            @test saddle.intersection_number in [-1, 0, 1]

            # In general, if the intersection number is non-zero, it must contribute
            @test saddle.contributing == (saddle.intersection_number != 0)
        end
    end

end
