using StaticArrays

# Generic Flow Data Structures:
# Point 
mutable struct Point{D::Int}
    coords::SVector{D,ComplexF64}
    active::Bool
end

# Simplex
mutable struct Simplex{D::Int}
    vertices::SVector{Point{D},Int}
    active::Bool
end
