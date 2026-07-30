using Symbolics

export get_thimble!
"""
    get_thimble!(z, S, saddle_point, params; mesh_type)

Calculates the Lefschetz thimble for a given saddle point using symbolic expressions.
This is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_thimble!`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The struct containing the saddle point.
- `params::Dict`: The parameters for the thimble calculation.
- `mesh_type::String`: The type of mesh to generate in 2D.

# Returns
- `Nothing`
"""
function get_thimble!(
    z::Union{Num,AbstractVector{Num}}, S::Num,
    saddle_point::Types.Saddle,
    params::Dict;
    mesh_type::String
)::Nothing
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]

    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_thimble!(native_S, native_grad, native_hessian, saddle_point, params, mesh_type=mesh_type)
end

export get_thimbles
"""
    get_thimbles(z, S, params, domain; mesh_type)

Calculates the Lefschetz thimbles for a given domain using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_thimbles`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `params::Dict`: The parameters for the thimble calculation.
- `domain::Vector{RealDomain}`: The domain over which to calculate the thimbles.
- `mesh_type::String`: The type of mesh to generate in 2D.

# Returns
- `Tuple{Vector{<:FlowPoint},Vector{<:Simplex}}`:The result of the flow (a set of segments in 1D, or a list of quadrilaterals/triangles in 2D).
"""
function get_thimbles(
    z::Union{Num,AbstractVector{Num}}, S::Num,
    params::Dict,
    domain::Vector{RealDomain};
    mesh_type::String
)::Tuple{Vector{<:FlowPoint},Vector{<:Simplex}}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})

    return get_thimbles(native_S, native_grad, params, domain, mesh_type=mesh_type)
end

export get_thimbles
"""
    get_thimbles(z, S, params, domain, contributing; mesh_type)

Calculates the Lefschetz thimbles for a given domain using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_thimbles`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `params::Dict`: The parameters for the thimble calculation.
- `domain::Vector{RealDomain}`: The domain over which to calculate the thimbles.
- `contributing::Bool`: Whether to include only contributing points.
- `mesh_type::String`: The type of mesh to generate in 2D.

# Returns
- `Vector{Saddle}`: A vector of the saddle points, with their thimbles populated.
"""
function get_thimbles(z::Union{Num,AbstractVector{Num}}, S::Num,
    params::Dict, domain::Vector{ComplexDomain},
    contributing::Bool; mesh_type::String
)::Vector{Saddle}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]

    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_thimbles(native_S, native_grad, native_hessian, params, domain, contributing, mesh_type=mesh_type)
end

export get_thimble_boundary!
"""
    get_thimble_boundary!(z, S, saddle_point, params; mesh_type)

Calculates the boundary of the Lefschetz thimble for a given saddle point using symbolic expressions.
This is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_thimble_boundary!`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The struct containing the saddle point.
- `params::Dict`: The parameters for the thimble boundary calculation.
- `mesh_type::String`: The type of mesh to generate in 2D.

# Returns
- `Nothing`
"""
function get_thimble_boundary!(
    z::Union{Num,AbstractVector{Num}}, S::Num,
    saddle_point::Types.Saddle,
    params::Dict;
    mesh_type::String
)::Nothing
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_thimble_boundary!(native_S, native_grad, native_hessian, saddle_point, params, mesh_type=mesh_type)
end

export get_thimble_boundaries
"""
    get_thimble_boundaries(z, S, domain, params, contributing; mesh_type, check)

Calculates the boundaries of the Lefschetz thimbles for all saddles using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete documentation, 
see `get_thimble_boundaries`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the thimble boundaries.
- `params::Dict`: The parameters for the thimble calculation.
- `contributing::Bool`: Whether to include only contributing points.
- `mesh_type::String`: The type of mesh to generate in 2D.
- `check::Function`: A function for checking equality of critical points.

# Returns
- `Vector{Saddle}`: A vector of the saddle points and their thimble boundaries.
"""
function get_thimble_boundaries(
    z::Union{Num,AbstractVector{Num}}, S::Num,
    domain::Vector{ComplexDomain},
    params::Dict, contributing::Bool;
    mesh_type::String,
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Saddle}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_thimble_boundaries(native_S, native_grad, native_hessian, domain, params, contributing, mesh_type=mesh_type, check=check)
end
