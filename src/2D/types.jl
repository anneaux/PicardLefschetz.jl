module Types

export ComplexDomain, PointA, QuadC, TriangleC, LineSeg, QuadA, TriangleA

struct ComplexDomain
    min::ComplexF64
    max::ComplexF64 #Union{ComplexF64,Nothing}

    ComplexDomain(rmin::Real, rmax::Real, imin::Real, imax::Real) = new(rmin + imin * im, rmax + imax * im)

    ComplexDomain(min::ComplexF64, max::ComplexF64) = new(min, max)

    ComplexDomain() = new(zero(ComplexF64), zero(ComplexF64))

end

### POINT
mutable struct PointA{T}
    x::T
    y::T
    active::Bool

    PointA(x, y) = new(x, y, true)
end

struct QuadC ### a quadrilateral with coordinates
    points::Vector{PointA} # length = 4
end

struct TriangleC ### a triangle with coordinates
    points::Vector{PointA} # length = 3
end

mutable struct LineSeg
    s::Union{UndefInitializer,PointA{T}} where T<:Union{<:Number,Vector{<:Number}}
    e::Union{UndefInitializer,PointA{T}} where T<:Union{<:Number,Vector{<:Number}}
    active::Bool
    startindex::Union{UndefInitializer,Int}
    endindex::Union{UndefInitializer,Int}

    LineSeg(s::PointA, e::PointA) = new(s, e, s.active && e.active, undef, undef)
    LineSeg(s::PointA, e::PointA, active::Bool) = new(s, e, active, undef, undef)
    LineSeg(sindex::Int, eindex::Int, active::Bool) = new(undef, undef, active, sindex, eindex)
    LineSeg(sindex::Int, eindex::Int) = new(undef, undef, false, sindex, eindex)
end

mutable struct QuadA
    indices::Vector{Int} ### this could be an MVector
    active::Bool
    QuadA(indices::Vector{Int64}) = new(indices, true)
end

mutable struct TriangleA
    indices::MVector{3,Int}
    active::Bool
    TriangleA(indices::AbstractVector{T}) where T<:Integer = new(indices, true)
    TriangleA(indices::AbstractVector{T}, a::Bool) where T<:Integer = new(indices, a)

end

end