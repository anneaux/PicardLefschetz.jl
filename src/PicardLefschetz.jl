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

# 1D code
include("1D/methods-1D.jl")

# 2D code
include("2D/downward-flow-quadrilaterals.jl")
include("2D/downward-flow-triangles.jl")
include("2D/flow-utils.jl")
include("2D/integrate-quads.jl")
include("2D/integrate-triangles.jl")
include("2D/necklace-for-dual-thimble.jl")
include("2D/PLIntegration2D.jl")
include("2D/saddle-point-method.jl")
include("2D/saddles-contributing.jl")
include("2D/saddles-generic.jl")
include("2D/thimble-from-saddle-point-quadrilaterals.jl")
include("2D/thimble-from-saddle-point-triangles.jl")
include("2D/utils.jl")

export Methods1D

end # PicardLefschetz
