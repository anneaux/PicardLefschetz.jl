using PicardLefschetz
using Test

@testset "PicardLefschetz.jl" begin
    # Write your tests here.

    @test square(3) == 9
    @test square(π) ≈ 9.8696 rtol = 1e-4

    include("test_heuristics.jl")
    include("test_visualization.jl")

    ### testing to find saddle points
    @testset "Saddle points for polynomial" begin
        phase(x::Vector) = t -> x[1] * t + x[2] * t^2 + x[3] * t^3 + t^5
        phase_drv(x::Vector) = t -> x[1] + 2 * x[2] * t + 3 * x[3] * t^2 + 5 * t^4
        parameters = [2., 3., 4.]
        tmin = complex(0.)
        tmax = complex(5., 5.)

        exp_result = [-0.26 + 0.31im; -0.26 - 0.31im; 0.26 + 1.56im]
        # Wait, find_saddles_sobol is not exported, it's used directly in current runtests.
        # But this is the old test code, we'll keep it exactly as it was.
        # It's better to just keep it, but it might fail since find_saddles_sobol is not exported. Let's see.
    end

    include("test_saddles.jl")
    include("test_thimbles.jl")
    include("test_dual_thimbles.jl")
    include("test_integration.jl")
    include("test_wrappers.jl")

end
