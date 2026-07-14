using Test
using PicardLefschetz
using StaticArrays
using LinearAlgebra

@testset "Gradient Descent & Flow" begin
    # Simple harmonic action: S(z) = 0.5 * z^2 (where z is 1D SVector{1, ComplexF64})
    # S_gradient(z) = z
    # S_hessian(z) = [1.0;;]
    S(z) = 0.5 * z[1]^2
    S_gradient(z) = SVector{1,ComplexF64}(z[1])
    S_hessian(z) = SMatrix{1,1,ComplexF64}(1.0)

    # 1. Test flow_vector_field!
    @testset "flow_vector_field!" begin
        dw = zeros(2)
        w = [2.0, 3.0] # z = 2.0 + 3.0im
        # descent: dz/dt = -conj(z) = -(2 - 3im) = -2 + 3im
        # dw[1] = -2.0, dw[2] = 3.0
        PicardLefschetz.GradientDescent.flow_vector_field!(dw, w, S_gradient, :descent)
        @test dw ≈ [-2.0, 3.0]

        # ascent: dz/dt = conj(z) = 2 - 3im
        # dw[1] = 2.0, dw[2] = -3.0
        PicardLefschetz.GradientDescent.flow_vector_field!(dw, w, S_gradient, :ascent)
        @test dw ≈ [2.0, -3.0]
    end

    # 2. Test flow_jacobian!
    @testset "flow_jacobian!" begin
        J = zeros(2, 2)
        w = [2.0, 3.0]
        # descent: J = [-A B; B A] where H = A + iB = 1.0 + 0im.
        # A = 1.0, B = 0.0
        # J = [-1.0 0.0; 0.0 1.0]
        PicardLefschetz.GradientDescent.flow_jacobian!(J, w, S_hessian, :descent)
        @test J ≈ [-1.0 0.0; 0.0 1.0]

        # ascent: J = [A -B; -B -A] = [1.0 0.0; 0.0 -1.0]
        PicardLefschetz.GradientDescent.flow_jacobian!(J, w, S_hessian, :ascent)
        @test J ≈ [1.0 0.0; 0.0 -1.0]
    end

    # 3. Test flow_single_point
    @testset "flow_single_point" begin
        # Start at z = 2.0 + 0.1im (active point)
        pt = Types.Point(SVector{1,ComplexF64}(2.0 + 0.1im))

        # Descent flow equations:
        # dx/dt = -x => x(t) = x0 * e^{-t}
        # dy/dt = y => y(t) = y0 * e^{t}
        # S(z) = 0.5 * (x^2 - y^2) + i * x * y
        # Real(S(z)) = 0.5 * (x^2 - y^2)
        # If we take 10 steps of size 0.05, t = 0.5.
        # expected final coords: x ≈ 2.0 * e^{-0.5} ≈ 1.213, y ≈ 0.1 * e^{0.5} ≈ 0.165
        coords_f, active = PicardLefschetz.GradientDescent.flow_single_point(
            pt, S, S_gradient, S_hessian, :descent, 0.0, 10, 0.05
        )
        @test coords_f[1] ≈ 2.0 * exp(-0.5) + im * 0.1 * exp(0.5) rtol = 1e-4
        @test active == true

        # Test threshold cutoff: if h_threshold = 1.0
        # real(S(z)) = 0.5 * (x^2 - y^2). Initially = 0.5 * (4 - 0.01) = 1.995.
        # When x falls below sqrt(2) ≈ 1.414, real(S(z)) <= 1.0.
        # This should stop integration.
        coords_f2, active2 = PicardLefschetz.GradientDescent.flow_single_point(
            pt, S, S_gradient, S_hessian, :descent, 1.0, 100, 0.05
        )
        @test active2 == false
        @test real(S(coords_f2)) <= 1.0
    end

    # 4. Test flow_points!
    @testset "flow_points!" begin
        pts = [
            Types.Point(SVector{1,ComplexF64}(2.0 + 0.1im), true),
            Types.Point(SVector{1,ComplexF64}(2.0 + 0.1im), false) # inactive should not flow
        ]
        PicardLefschetz.GradientDescent.flow_points!(
            pts, S, S_gradient, S_hessian, :descent, 0.0, 10, 0.05
        )
        @test pts[1].coords[1] ≈ 2.0 * exp(-0.5) + im * 0.1 * exp(0.5) rtol = 1e-4
        @test pts[1].active == true
        # Inactive point should remain unchanged
        @test pts[2].coords[1] == 2.0 + 0.1im
        @test pts[2].active == false
    end

    # 5. Efficiency check: allocation tests
    @testset "Zero heap allocations inside inner integration loop" begin
        dw = zeros(2)
        w = [2.0, 3.0]
        allocs_vf = @allocated PicardLefschetz.GradientDescent.flow_vector_field!(dw, w, S_gradient, :descent, Val(1))
        allocs_jac = @allocated PicardLefschetz.GradientDescent.flow_jacobian!(zeros(2, 2), w, S_hessian, :descent, Val(1))

        @test allocs_vf == 0
        @test allocs_jac == 0
    end
end
