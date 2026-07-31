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
using SimplexQuad

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

include("core/heuristics.jl")

export get_pl_heuristics_1d, get_pl_heuristics_2d, resolve_heuristics


end # PicardLefschetz
