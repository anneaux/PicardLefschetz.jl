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
    saddle_params_1d = Dict(
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
        saddles = find_numerical_saddles(phase_drv_1d(params_1d), domain_1d, saddle_params_1d)
        @test saddles isa Vector
        # We expect up to 4 saddles for a degree 4 derivative
        @test length(saddles) <= 4
    end

    @testset "1D Analytic Saddles" begin
        # Provide an initial guess close to origin
        saddle = find_analytic_saddles(phase_drv_1d(params_1d), [0.0 + 0.0im], 5)
        @test saddle isa Vector
    end

    @testset "1D Contribution Check" begin
        saddles = find_numerical_saddles(phase_drv_1d(params_1d), domain_1d, saddle_params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            check_contribution!(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), saddle, domain_1d[1], saddle_params_1d)
            @test saddle.contributing isa Bool
        end
    end

    @testset "2D Numerical Saddles" begin
        saddles = find_numerical_saddles(phase_drv_2d(params_2d), domain_2d, saddle_params_2d)
        @test saddles isa Vector
    end

    @testset "2D Analytic Saddles" begin
        # Provide an initial guess close to origin
        saddle = find_analytic_saddles(phase_drv_2d(params_2d), [0.1 + 0.1im, 0.1 - 0.1im], 5)
        @test saddle isa Tuple
    end

    @testset "2D Contribution Check" begin
        saddles = find_numerical_saddles(phase_drv_2d(params_2d), domain_2d, saddle_params_2d)
        if !isempty(saddles)
            saddle = saddles[1]
            check_contribution!(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), saddle, domain_2d[1], saddle_params_2d)
            @test saddle.contributing isa Bool
        end
    end

end
