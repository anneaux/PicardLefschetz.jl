using Test
using PicardLefschetz
using StaticArrays
using LinearAlgebra

@testset "Lefschetz Thimble Contour Mesh" begin
    # 1. calculate_eigenvectors!
    @testset "calculate_eigenvectors!" begin
        # Action S(z) = 0.5 * (z_1^2 - z_2^2)
        # Hessian: S_xx = 1, S_yy = -1 (constant)
        # We compute eigenvalues of Hessian.
        # Let's check saddle point at (0, 0)
        saddle = Types.Point(SVector{2,ComplexF64}(0.0, 0.0))
        f(z) = 0.5 * (z[1]^2 - z[2]^2)

        pos_eigs = Types.Point{2}[]
        neg_eigs = Types.Point{2}[]

        PicardLefschetz.LefschetzThimbleContour.calculate_eigenvectors!(
            saddle, f, pos_eigs, neg_eigs
        )

        @test length(pos_eigs) == 2
        @test length(neg_eigs) == 2

        # Verify that they are sorted/grouped correctly.
        # The eigenvalues of the imaginary part of Hessian should classify the eigenvectors.
    end

    # 2. _generate_boundary_mesh! (Cross-Polytope, D < 4)
    @testset "generate_boundary_mesh! (D < 4)" begin
        saddle = Types.Point(SVector{2,ComplexF64}(0.0, 0.0))
        ϵ = 0.1
        eigenvectors = [
            Types.Point(SVector{2,ComplexF64}(1.0, 0.0)),
            Types.Point(SVector{2,ComplexF64}(0.0, 1.0))
        ]

        points, simplices = PicardLefschetz.LefschetzThimbleContour.generate_boundary_mesh!(
            saddle, ϵ, eigenvectors
        )

        # For D=2 < 4, it uses Cross-Polytope.
        # The number of vertices should be 2*D = 4.
        @test length(points) == 4

        # Coordinates should be saddle ± ϵ * eigenvector
        expected_coords = [
            SVector{2,ComplexF64}(0.1, 0.0),
            SVector{2,ComplexF64}(-0.1, 0.0),
            SVector{2,ComplexF64}(0.0, 0.1),
            SVector{2,ComplexF64}(0.0, -0.1)
        ]

        for ec in expected_coords
            @test any(p.coords ≈ ec for p in points)
        end
    end

    # 3. Smolyak Helpers
    @testset "Smolyak Helpers" begin
        # generate_clenshaw_curtis_nodes
        nodes_1 = PicardLefschetz.LefschetzThimbleContour.generate_clenshaw_curtis_nodes(1)
        @test nodes_1 == [0.0]

        nodes_2 = PicardLefschetz.LefschetzThimbleContour.generate_clenshaw_curtis_nodes(2)
        @test nodes_2 ≈ [-1.0, 0.0, 1.0]

        # map_to_sphere
        # For M=3 (sphere in 4D space)
        x = SVector{3,Float64}(0.0, 0.0, 0.0)
        u = PicardLefschetz.LefschetzThimbleContour.map_to_sphere(x)
        @test length(u) == 4
        @test norm(u) ≈ 1.0

        # get_smolyak_grid_point_combinations
        grid = PicardLefschetz.LefschetzThimbleContour.get_smolyak_grid_point_combinations(3, 2)
        @test length(grid) > 0
        @test all(norm(PicardLefschetz.LefschetzThimbleContour.map_to_sphere(g)) ≈ 1.0 for g in grid)
    end

    # 4. _generate_boundary_mesh! (Smolyak, D >= 4)
    @testset "generate_boundary_mesh! (D >= 4)" begin
        saddle = Types.Point(SVector{4,ComplexF64}(0.0, 0.0, 0.0, 0.0))
        ϵ = 0.05
        eigenvectors = [
            Types.Point(SVector{4,ComplexF64}(1.0, 0.0, 0.0, 0.0)),
            Types.Point(SVector{4,ComplexF64}(0.0, 1.0, 0.0, 0.0)),
            Types.Point(SVector{4,ComplexF64}(0.0, 0.0, 1.0, 0.0)),
            Types.Point(SVector{4,ComplexF64}(0.0, 0.0, 0.0, 1.0))
        ]

        points = Types.Point{4}[]
        PicardLefschetz.LefschetzThimbleContour.generate_boundary_mesh!(
            points, saddle, ϵ, eigenvectors
        )

        # For D=4 >= 4, it uses Smolyak sparse grid.
        @test length(points) > 0

        # Verify that all generated coordinates lie on a sphere of radius ϵ around the saddle point
        for p in points
            dist = norm(p.coords - saddle.coords)
            @test dist ≈ ϵ
        end
    end
end
