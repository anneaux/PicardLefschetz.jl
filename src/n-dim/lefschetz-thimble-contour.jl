using StaticArrays
using FiniteDiff

module LefschetzThimbleContour

using ..Types: Point
using ..Types: Simplex

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
    positive_eigenvectors::Vector{Point{D}},
    negative_eigenvectors::Vector{Point{D}})::Nothing where {D}

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
    points::Vector{Point{D}},
    saddle_point::Point{D},
    ϵ::Float64,
    eigenvectors::Vector{Point{D}})::Nothing where {D}

    condition = D < 4
    return _generate_boundary_mesh!(Val(condition), points, saddle_point, ϵ, eigenvectors)

end

# Generating the boundary mesh, using Cross-Polytope algorithm. Only applicable if the complex
# dimension of the space is less than 4.
function _generate_boundary_mesh!(
    ::Val{true},
    points::Vector{Point{D}},
    saddle_point::Point{D},
    ϵ::Float64,
    eigenvectors::Vector{Point{D}})::Nothing where {D}
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

# Generating the boundary mesh, using Smolyak sparse grid. Only applicable if the complex
# dimension of the space is greater than or equal to 4.
function _generate_boundary_mesh!(
    ::Val{false},
    points::Vector{Point{D}},
    saddle_point::Point{D},
    ϵ::Float64,
    eigenvectors::Vector{Point{D}})::Nothing where {D}

    M = D - 1
    L = 2
    hypercube_grid = get_smolyak_grid_point_combinations(M, L)
    sphere_vertices = [map_to_sphere(x) for x in hypercube_grid]
    empty!(points)
    for vertex in sphere_vertices
        coordinate = saddle_point.coords + ϵ * sum(vertex[k] * eigenvectors[k].coords for k in 1:D)
        push!(points, Point{D}(coordinate, true))
    end

    return nothing
end

# Generate Clenshaw-Curtis nodes.
function generate_clenshaw_curtis_nodes(i::Int)::Vector{Float64}
    if i == 1
        return [0.0]
    else
        n = 2^(i - 1) + 1
        return [-cos(π * k / (n - 1)) for k in 0:(n-1)]
    end
end

# Collect the Smolyak grid point combinations in [-1, 1]^D.
function get_smolyak_grid_point_combinations(M::Int, L::Int)::Vector{SVector{M,Float64}}
    grid = Set{SVector{M,Float64}}()
    iterate_product(arrays) = vec(collect(Base.product(arrays...)))
    generate_indices!(Int[], L, grid, iterate_product)
    return collect(grid)
end

# Generate the indices for the Smolyak sparse grid combinations.
function generate_indices!(
    current_index::Vector{Int},
    L::Int,
    grid::Set{SVector{M,Float64}},
    iterate_product::Function)::Nothing where {M}

    if length(current_index) == M
        if sum(current_index) <= M + L - 1
            nodes_1d = [generate_clenshaw_curtis_nodes(index) for index in current_index]

            for element in iterate_product(nodes_1d)
                push!(grid, SVector{M,Float64}(element))
            end
        end
        return
    end

    for level in 1:(L+1)
        push!(current_index, level)
        if sum(current_index) <= M + L - 1 + (M - length(current_index))
            generate_indices!(current_index, M, L, iterate_product)
        end
        pop!(current_index)
    end
end

# Map a point in [-1, 1]^(D - 1) to angles, then to Cartesian coordinates on S^(D - 1) in R^D
function map_to_sphere(x::SVector{M,Float64})::Vector{Float64} where {M}
    angles = Vector{Float64}(undef, M)
    for i in 1:(M-1)
        angles[i] = π * (x[i] + 1) / 2
    end
    angles[M] = (x[M] + 1) * π

    D = M + 1
    u = zeros(D)
    sin_accum = 1.0
    for i in 1:(D-1)
        u[i] = sin_accum * cos(angles[i])
        sin_accum *= sin(angles[i])
    end
    u[D] = sin_accum
    return SVector{D,Float64}(u)
end

end