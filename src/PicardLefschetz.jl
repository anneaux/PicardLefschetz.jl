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

include("core/saddles.jl")
include("core/integration.jl")
include("core/thimbles.jl")
include("core/dual-thimbles.jl")

using .DualThimble, .Integration, .Saddle, .Thimble

export Types, DualThimble, Integration, Saddle, Thimble

end # PicardLefschetz