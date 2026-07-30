module Methods2D

using LinearAlgebra
using FiniteDiff
using StaticArrays
using Contour
using ..Types

include("Utils/utils.jl")
include("necklace-for-dual-thimble.jl")
include("Quadrilaterals/quadrilaterals.jl")
include("Triangles/triangles.jl")
include("Saddle/saddle-point-method.jl")

export Utils, Quadrilateral, Triangles, SaddlePoint, DualThimble

end