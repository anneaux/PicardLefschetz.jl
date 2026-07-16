module Methods1D

include("line-intersection.jl")
include("path-flow.jl")
include("integration.jl")
include("critical-points.jl")
include("saddle-point-method.jl")

export PathFlow, LineIntersection, CriticalPoints, SaddlePoint, Integration

end