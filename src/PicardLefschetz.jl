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
include("2D/methods-2D.jl")

export Methods1D, Methods2D

end # PicardLefschetz
