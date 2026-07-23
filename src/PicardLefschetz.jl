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

# Types
include("core/types.jl")

# 1D code
include("core/1D/methods-1D.jl")

# 2D code
include("core/2D/methods-2D.jl")

using Types
using Methods1D
using Methods2D

include("core/dual-thimbles.jl")
include("core/integration.jl")
include("core/saddles.jl")
include("core/thimbles.jl")

using .DualThimble, .Integration, .Saddle, .Thimble

export Types, DualThimble, Integration, Saddle, Thimble

end # PicardLefschetz