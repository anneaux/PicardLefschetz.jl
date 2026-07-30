module Types

using StaticArrays
import Base: isequal, hash, getindex

export FlowPoint, Line, Simplex, ComplexDomain, RealDomain, Saddle

mutable struct FlowPoint{N,T<:Number}
    coords::MVector{N,T}
    active::Bool

    FlowPoint(z::Complex) = new{1,typeof(z)}(@MVector([z]), true)
    FlowPoint(z1::Complex, z2::Complex) = new{2,typeof(z1)}(@MVector([z1, z2]), true)

end

Base.isequal(p1::FlowPoint, p2::FlowPoint) = isequal(p1.coords, p2.coords)
Base.hash(p::FlowPoint, h::UInt) = hash(p.coords, h)
Base.getindex(p::FlowPoint, i::Int) = p.coords[i]

mutable struct Simplex{N,T}
    vertices::MVector{N,T}
    active::Bool

    Simplex{N,T}(verts::AbstractVector{T}, active::Bool=true) where {N,T} = new{N,T}(MVector{N,T}(verts), active)
    Simplex{N,T}(verts...) where {N,T} = new{N,T}(MVector{N,T}(verts...), true)
end

mutable struct Line
    start_point::SVector{2,Float64}
    end_point::SVector{2,Float64}

    Line(t1::Tuple{Float64,Float64}, t2::Tuple{Float64,Float64}) = new(SA[t1...], SA[t2...])
end

struct ComplexDomain
    min::ComplexF64
    max::ComplexF64

    ComplexDomain(real_min::Real, real_max::Real, imag_min::Real, imag_max::Real) = new(real_min + imag_min * 1im, real_max + imag_max * 1im)
    ComplexDomain(min::ComplexF64, max::ComplexF64) = new(min, max)
    ComplexDomain() = new(zero(ComplexF64), zero(ComplexF64))
end

struct RealDomain
    min::Float64
    max::Float64

    RealDomain(min::Real, max::Real) = new(Float64(min), Float64(max))
    RealDomain(min::Real, max::Real, ::Val{:empty}) = new(Inf, -Inf)
    RealDomain() = new(zero(Float64), zero(Float64))
end

mutable struct Saddle{T,TB,DT,DTB}
    saddle::FlowPoint
    contributing::Union{Nothing,Bool}
    intersection_number::Union{Nothing,Int}
    thimble::Union{Nothing,T}
    thimble_boundary::Union{Nothing,TB}
    dual_thimble::Union{Nothing,DT}
    dual_thimble_boundary::Union{Nothing,DTB}
    integral::Union{Nothing,ComplexF64,AbstractVector{ComplexF64}}
end

function Saddle(; saddle::FlowPoint,
    contributing::Union{Nothing,Bool}=nothing,
    intersection_number::Union{Nothing,Int}=nothing,
    thimble=nothing,
    thimble_boundary=nothing,
    dual_thimble=nothing,
    dual_thimble_boundary=nothing,
    integral::Union{Nothing,ComplexF64,AbstractVector{ComplexF64}}=nothing
)
    T = thimble === nothing ? Any : typeof(thimble)
    TB = thimble_boundary === nothing ? Any : typeof(thimble_boundary)
    DT = dual_thimble === nothing ? Any : typeof(dual_thimble)
    DTB = dual_thimble_boundary === nothing ? Any : typeof(dual_thimble_boundary)
    return Saddle{T,TB,DT,DTB}(saddle, contributing, intersection_number, thimble, thimble_boundary, dual_thimble, dual_thimble_boundary, integral)
end

Saddle{T}(; kwargs...) where T = Saddle{T,T,T,T}(; kwargs...)

Base.length(p::FlowPoint{N}) where {N} = N
Base.length(s::Saddle) = length(s.saddle)
Base.getindex(s::Saddle, i::Int) = s.saddle.coords[i]

function convert(saddle::Saddle)
    thimble_mesh = convert_to_mesh(saddle.thimble)
    thimble_boundary_mesh = convert_to_mesh(saddle.thimble_boundary)
    dual_thimble_mesh = convert_to_mesh(saddle.dual_thimble)
    dual_thimble_boundary_mesh = convert_to_mesh(saddle.dual_thimble_boundary)

    return Saddle(
        saddle=saddle.saddle,
        contributing=saddle.contributing,
        intersection_number=saddle.intersection_number,
        thimble=thimble_mesh,
        thimble_boundary=thimble_boundary_mesh,
        dual_thimble=dual_thimble_mesh,
        dual_thimble_boundary=dual_thimble_boundary_mesh,
        integral=saddle.integral
    )
end

function convert_to_mesh(point_cloud::Tuple{Vector{<:FlowPoint},Vector{Simplex{A,Int}}}) where A
    points, simplices = point_cloud
    return [Simplex{A,FlowPoint}(points[simplex.vertices]) for simplex in simplices]
end

convert_to_mesh(::Nothing) = nothing
convert_to_mesh(point_cloud::Vector{<:FlowPoint}) = point_cloud
convert_to_mesh(point_cloud::Vector{<:AbstractVector}) = point_cloud

end