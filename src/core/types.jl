module Types

using StaticArrays

export FlowPoint, Line, Simplex, ComplexDomain, RealDomain, Saddle

mutable struct FlowPoint{N,T<:Number}
    coords::MVector{N,T}
    active::Bool

    FlowPoint(z::Complex) = new{1,typeof(z)}(@MVector([z]), true)
    FlowPoint(z1::Complex, z2::Complex) = new{2,typeof(z1)}(@MVector([z1, z2]), true)

end

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

Base.@kwdef mutable struct Saddle{T}
    saddle::Vector{FlowPoint}
    contributing::Union{Nothing,Bool} = nothing
    thimble::Union{Nothing,T} = nothing
    thimble_boundary::Union{Nothing,T} = nothing
    dual_thimble::Union{Nothing,T} = nothing
    dual_thimble_boundary::Union{Nothing,T} = nothing
    integral::Union{Nothing,ComplexF64} = nothing
end

end