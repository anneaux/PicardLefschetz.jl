using PicardLefschetz
using Test
using LinearAlgebra
using GLMakie

GLMakie.activate!()

@testset "Visualization Tests" begin
    S_quad(z) = z^2
    drv_quad(z) = 2z
    ξ = 0.0 + 0.0im
    ω = 1.0
    
    # Test 1: record_thimble_flow_animation
    # We'll write to a temporary file to avoid cluttering the repository
    tmp_mp4 = tempname() * ".mp4"
    try
        # Run animation using the new heuristics interface
        result_file = record_thimble_flow_animation(S_quad, drv_quad, ξ, ω, -2.0, 2.0, tmp_mp4; preset=:fast, keep_connected=true)
        @test isfile(result_file)
        @test result_file == tmp_mp4
    finally
        isfile(tmp_mp4) && rm(tmp_mp4)
    end
    
    # Test 2: run_and_plot
    fig = Figure()
    ax = Axis(fig[1, 1])
    (points, simplices) = PicardLefschetz.initialise(-2.0, 2.0, 10.0)
    # Verify run_and_plot doesn't error out
    @test run_and_plot(ax, S_quad, points, simplices; title="Test Plot", saddles=[ξ]) isa Any

    # Test 3: plot_thimble_3d
    # Mocking simplices (triangles)
    mock_p1 = (1.0, 2.0)
    mock_p2 = (3.0, 4.0)
    mock_p3 = (5.0, 6.0)
    mock_tris = [(points = [mock_p1, mock_p2, mock_p3],)]
    coords_3d_mock(p) = (p[1], p[2], p[1] + p[2])

    fig3d = Figure()
    ax3d = LScene(fig3d[1, 1])
    @test plot_thimble_3d(ax3d, mock_tris, coords_3d_mock; saddles=[mock_p1]) isa Any
end
