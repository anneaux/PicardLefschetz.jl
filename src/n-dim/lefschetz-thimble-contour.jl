using StaticArrays
using FiniteDiff

module LefschetzThimbleContour

# Generic Flow Data Structures:
# Point 
export Point
"""
A mutable struct representing a point in D-dimensional space.

@param D::Int The complex dimension of the point.
@param coords::SVector{D,ComplexF64} The coordinates of the point in a Riemann chart.
@param active::Bool Whether the point is active or not. (Criteria depends on parameters given initially.)
"""
mutable struct Point{D::Int}
    coords::SVector{D,ComplexF64}
    active::Bool
end

Point(coords::SVector{D,ComplexF64}) = Point{D}(coords, true);

# Simplex
export Simplex
"""
A mutable struct representing a simplex in D-dimensional space.

@param D::Int The complex dimension of the simplex.
@param vertices::SVector{D,Point} The vertices of the simplex.
@param active::Bool Whether the simplex is active or not.
"""
mutable struct Simplex{D::Int}
    vertices::SVector{D,Point}
    active::Bool
end

Simplex(vertices::SVector{D,Point}) = Simplex{D}(vertices, true);

# Calculating Eigenvectors of Hessian for a given saddle point
export calculate_eigenvectors!
"""
Calculates the eigenvectors of the Hessian matrix for a given saddle point.

@param D::Int The complex dimension of the saddle point.
@param saddle_point::Point{D} The saddle point.
@param f::Function The function to calculate the Hessian of.
@param positive_eigenvectors::Vector{Point} The positive eigenvectors.
@param negative_eigenvectors::Vector{Point} The negative eigenvectors.

@return Nothing. Modifies the positive_eigenvectors and negative_eigenvectors in place.
"""
function calculate_eigenvectors!(
    saddle_point::Point{D},
    f::Function,
    positive_eigenvectors::Vector{Point},
    negative_eigenvectors::Vector{Point}) where {D}

    # Calculate Hessian using finite difference method. Convert input to a real vector
    # where the real vector has entries for each real and imaginary component. Then,
    # calculate the new complex vector using that real vector, and then evaluate f over
    # said complex vector. 
    X = Vector{Float64}(undef, 2 * D)
    for i in 1:D
        X[2i-1] = real(saddle_point.coords[i])
        X[2i] = imag(saddle_point.coords[i])
    end
    hessian = FiniteDiff.finite_difference_hessian(
        tvec -> imag(f(SVector{D,ComplexF64}(ntuple(index -> complex(tvec[2*index-1], tvec[2*index]), D)))),
        X
    )

    # Decompose eigenvectors, and sort them by whether they're positive or negative. 
    # Each pair (real and complex) are entered into the list of eigenvectors, which
    # are then reconstructed into the complex eigenvectors.
    eigenvector_decomposition = eigen(hessian)
    eigenvalues = eigenvector_decomposition.values
    eigenvectors = eigenvector_decomposition.vectors
    pairs = [(eigenvalues[i], eigenvectors[:, i]) for i in 1:2*D]
    sort!(pairs, by=x -> x[1])

    for i in 1:D
        vector = pairs[i][2]
        complex_vector = SVector{D,ComplexF64}(ntuple(index -> complex(vector[2*index-1], vector[2*index]), D))
        push!(negative_eigenvectors, Point(complex_vector))
    end

    for i in (D+1):(2*D)
        vector = pairs[i][2]
        complex_vector = SVector{D,ComplexF64}(ntuple(index -> complex(vector[2*index-1], vector[2*index]), D))
        push!(positive_eigenvectors, Point(complex_vector))
    end
end

# Generating the boundary mesh, depending on the complex dimension of the space.
# For dims 1 - 3, it uses the Cross-Polytope algorithm. For dims 4 - 4+ it uses the
# Smolyak sparse grid.
export generate_boundary_mesh!
"""
Generates a boundary mesh. If the complex dimension is less than 4, it uses the
Cross-Polytope algorithm. For dimension greater than or equal to 4, it uses the 
Smolyak sparse grid.

@param points::Vector{Point} The initial points for the boundary mesh.
@param saddle_point::Point{D} The saddle point.
@param ϵ::Float64 The radius of the initial sphere.
@param eigenvectors::Vector{Point} The eigenvectors.

@return Nothing. It mutates the points vector in place.
"""
function generate_boundary_mesh!(
    points::Vector{Point},
    saddle_point::Point{D},
    ϵ::Float64,
    eigenvectors::Vector{Point}) where {D}

    # Generate vertices of a D - 1 simplex as unit vectors in the basis directions.
    vertices = Vector{SVector{D,Float64}}()
    for d in 1:D
        position_vector = zeros(D)
        position_vector[d] = 1.0
        push!(vertices, SVector{D,Float64}(position_vector))
        push!(vertices, SVector{D,Float64}(-position_vector))
    end

    # Generate simplices using the different coordinate axis.
    simplices = Vector{Simplex{D}}()
    for bits in 0:(2^D-1)
        vertices_indices = Int[]
        for d in 1:D
            bit = (bits >> (d - 1)) & 1
            push!(vertices_indices, 2 * d - 1 + bit)
        end
        push!(simplices, Simplex{D}(SVector{D,Point}(vertices[vertices_indices]), true))
    end

    # Generate boundary mesh by mapping generated simplices to the local coordinate chart about the saddle point.
    points = Vector{Point{D}}()
    for vertex in vertices
        coordinate = saddle_point + ϵ * (sum(vertex[k] * eigenvectors[k] for k in 1:D))
        push!(points, Point{D}(coordinate, true))
    end

end
