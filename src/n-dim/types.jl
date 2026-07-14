module Types
# Generic Flow Data Structures:
# Point 
export Point
"""
A mutable struct representing a point in D-dimensional space.

@param D::Int The complex dimension of the point.
@param coords::SVector{D,ComplexF64} The coordinates of the point in a Riemann chart.
@param active::Bool Whether the point is active or not. (Criteria depends on parameters given initially.)
"""
mutable struct Point{D}
    coords::SVector{D,ComplexF64}
    active::Bool
end

Point(coords::SVector{D,ComplexF64}) = Point{D}(coords, true)

# Simplex
export Simplex
"""
A mutable struct representing a simplex in D-dimensional space.

@param D::Int The complex dimension of the simplex.
@param vertices::SVector{D,Point} The vertices of the simplex.
@param active::Bool Whether the simplex is active or not.
"""
mutable struct Simplex{D}
    vertices::SVector{D,Point}
    active::Bool
end

Simplex(vertices::SVector{D,Point}) = Simplex{D}(vertices, true)

end