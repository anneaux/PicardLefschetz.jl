module Types

using StaticArrays

export MyPoint, Index, Line2

mutable struct MyPoint
    coord::Complex
    active::Bool
    MyPoint(coord) = new(coord, true)
end

mutable struct Index
    coord::Vector{Int}
    active::Bool
    Index(coord) = new(coord, true)
end

struct Line2
    s::StaticArrays.SVector{2,Float64}
    e::StaticArrays.SVector{2,Float64}

    Line2(t1::Tuple{Float64,Float64}, t2::Tuple{Float64,Float64}) =
        new(SA[t1...], SA[t2...])
end

end