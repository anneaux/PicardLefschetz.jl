module PicardLefschetz

### make sure they're added to the Project.toml by doing ] activate .,  add PackageName
using LinearAlgebra
### for finding critical points
using NLsolve
using Sobol
### for the contour intersection
using Contour
using StaticArrays
using GeometryBasics
using FiniteDiff
using FastGaussQuadrature

# Types
include("core/types.jl")

# 1D code
include("core/1D/methods1D.jl")

# 2D code
include("core/2D/methods2D.jl")

include("core/saddles.jl")
include("core/integration.jl")
include("core/thimbles.jl")
include("core/dual-thimbles.jl")

using .DualThimble, .Integration, .Saddle, .Thimble

export Types, DualThimble, Integration, Saddle, Thimble
include("1D/integration-path-flow.jl")
include("1D/heuristics.jl")

include("1D/critical-points.jl")
include("1D/line-intersection.jl")
include("1D/saddle-point-method.jl")
include("1D/visualization.jl")

export dissect_inactive_segments, has_adjacent_inactive_segments, get_pl_heuristics_1d, get_pl_heuristics_2d, resolve_heuristics, run_and_plot, record_thimble_flow_animation, plot_thimble_3d, get_thimble, integrate_thimble, get_directional_r_osc, ballradius


end # PicardLefschetz
