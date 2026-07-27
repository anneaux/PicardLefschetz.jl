using Test
using PicardLefschetz
using PicardLefschetz.Types
using PicardLefschetz.Integration
using PicardLefschetz.Saddle

@testset "Integration Core Methods" begin

    # 1D Setup
    phase_1d(x::Vector) = t -> x[1] * t + x[2] * t^2 + x[3] * t^3 + t^5
    phase_drv_1d(x::Vector) = t -> x[1] + 2 * x[2] * t + 3 * x[3] * t^2 + 5 * t^4
    phase_hess_1d(x::Vector) = t -> 2 * x[2] + 6 * x[3] * t + 20 * t^3
    prefactor_1d(t) = 1.0 + 0im
    params_1d = [2., 3., 4.]
    domain_1d = [RealDomain(-5.0, 5.0)]
    complex_domain_1d = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]
    integ_params_1d = Dict(
        "point_count" => 10,
        "accuracy" => 5,
        "grid_resolution" => 10,
        "flow_steps" => 10,
        "grid_spacing" => 0.1,
        "gradient_normalisation_threshold" => 1e-4,
        "flow_step_factor" => 0.1,
        "subdivision_threshold" => 0.1,
        "height_threshold" => 0.01,
        "max_flow_steps" => 10,
        "max_grid_element_count" => 100,
        "integral_accuracy" => 1e-4,
        "integral_relative_error" => 1e-4,
        "verbose" => false,
        "output_dim" => 1
    )

    # 2D Setup
    phase_2d(x::Vector) = t -> x[1] * t[1]^2 + x[2] * t[2]^2 + t[1] * t[2]
    phase_drv_2d(x::Vector) = t -> [2 * x[1] * t[1] + t[2], 2 * x[2] * t[2] + t[1]]
    phase_hess_2d(x::Vector) = t -> [2*x[1] 1; 1 2*x[2]]
    prefactor_2d(t) = 1.0 + 0im
    params_2d = [1.0, 1.0]
    domain_2d = [RealDomain(-5.0, 5.0), RealDomain(-5.0, 5.0)]
    complex_domain_2d = [ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im), ComplexDomain(-5.0 - 5.0im, 5.0 + 5.0im)]
    integ_params_2d = Dict(
        "point_count" => 10,
        "accuracy" => 5,
        "grid_resolution" => 10,
        "flow_steps" => 10,
        "grid_spacing" => 0.1,
        "gradient_normalisation_threshold" => 1e-4,
        "flow_step_factor" => 0.1,
        "subdivision_threshold" => 0.1,
        "height_threshold" => 0.01,
        "max_grid_element_count" => 100,
        "integral_accuracy" => 1e-4,
        "integral_relative_error" => 1e-4,
        "verbose" => false,
        "output_dim" => 1,
        "GL_order" => 3,
        "simplex_order" => 3
    )

    prefactor_2d_multi(t) = [1.0 + 0im, 2.0 + 0im]
    integ_params_2d_multi = copy(integ_params_2d)
    integ_params_2d_multi["output_dim"] = 2

    @testset "1D integrate_thimble!" begin
        saddles = find_numerical_saddles(phase_drv_1d(params_1d), complex_domain_1d, integ_params_1d)
        if !isempty(saddles)
            saddle = saddles[1]
            integrate_thimble!(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), saddle, prefactor_1d)
            @test saddle.integral !== nothing
            @test saddle.integral isa ComplexF64
        end
    end

    @testset "1D integrate_thimbles (flow)" begin
        res = integrate_thimbles(phase_1d(params_1d), phase_drv_1d(params_1d), domain_1d, [0.0], prefactor_1d, integ_params_1d, "flow")
        # Just verifying it returns something
        @test res !== nothing
    end

    @testset "1D integrate_thimbles (SPM)" begin
        res = integrate_thimbles(phase_1d(params_1d), phase_drv_1d(params_1d), phase_hess_1d(params_1d), complex_domain_1d, integ_params_1d, prefactor_1d)
        @test res !== nothing
        @test res isa Vector{<:Types.Saddle}
    end

    @testset "2D integrate_thimble!" begin
        saddles = find_numerical_saddles(phase_drv_2d(params_2d), complex_domain_2d, integ_params_2d)
        if !isempty(saddles)
            saddle = saddles[1]
            integrate_thimble!(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), saddle, prefactor_2d)
            @test saddle.integral !== nothing
            @test saddle.integral isa ComplexF64

            saddle_multi = deepcopy(saddle)
            integrate_thimble!(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), saddle_multi, prefactor_2d_multi)
            @test saddle_multi.integral !== nothing
            @test saddle_multi.integral isa AbstractVector{ComplexF64}
        end
    end

    @testset "2D integrate_thimbles (fixed flow)" begin
        res = integrate_thimbles(phase_2d(params_2d), phase_drv_2d(params_2d), domain_2d, [0.0, 0.0], prefactor_2d, integ_params_2d, "fixed")
        @test res !== nothing
    end

    @testset "2D integrate_thimbles (SPM)" begin
        res = integrate_thimbles(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), complex_domain_2d, integ_params_2d, prefactor_2d)
        @test res !== nothing
        @test res isa Vector{<:Types.Saddle}

        res_multi = integrate_thimbles(phase_2d(params_2d), phase_drv_2d(params_2d), phase_hess_2d(params_2d), complex_domain_2d, integ_params_2d_multi, prefactor_2d_multi)
        @test res_multi !== nothing
        @test res_multi isa Vector{<:Types.Saddle}
    end
end
