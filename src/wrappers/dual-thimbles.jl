using Symbolics

export get_dual_thimble!
"""
    get_dual_thimble!(z, S, saddle_point, params)

Calculates the dual Lefschetz thimble for a given saddle point using symbolic expressions.
This is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_dual_thimble!`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The struct containing the saddle point.
- `params::Dict`: The parameters for the thimble calculation.

# Returns
- `Nothing`
"""
function get_dual_thimble!(
    z::Union{Num,AbstractVector{Num}}, S::Num,
    saddle_point::Saddle,
    params::Dict
)::Nothing
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimble!(native_S, native_grad, native_hessian, saddle_point, params)
end

export get_dual_thimbles
"""
    get_dual_thimbles(z, S, params, domain, contributing)

Calculates the dual Lefschetz thimbles for a given domain using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_dual_thimbles`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `params::Dict`: The parameters for the thimble calculation.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the thimbles.
- `contributing::Bool`: Whether to include only contributing points.

# Returns
- `Vector{Saddle}`: A vector of the saddle points, with their dual thimbles populated.
"""
function get_dual_thimbles(
    z::Union{Num, AbstractVector{Num}}, S::Num, params::Dict,
    domain::Vector{ComplexDomain},
    contributing::Bool
)::Vector{Saddle}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimbles(native_S, native_grad, native_hessian, params, domain, contributing)
end

export get_dual_thimble_boundary!
"""
    get_dual_thimble_boundary!(z, S, saddle_point, params)

Calculates the boundary of the dual thimble for a given saddle point using gradient ascent. This is a wrapper 
function for compatibility with the Symbolics.jl package. For complete documentation, see 
`get_dual_thimble_boundary!`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Saddle`: The struct containing the saddle point.
- `params::Dict`: The parameters for the dual thimble calculation.

# Returns
- `Nothing`
"""
function get_dual_thimble_boundary!(
    z::Union{Num, AbstractVector{Num}}, S::Num,
    saddle_point::Saddle,
    params::Dict
)::Nothing
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimble_boundary!(native_S, native_grad, native_hessian, saddle_point, params)
end

export get_dual_thimble_boundaries
"""
    get_dual_thimble_boundaries(z, S, domain, params, contributing)

Calculates the boundaries of the dual thimbles for all saddles using gradient ascent. This is a wrapper 
function for compatibility with the Symbolics.jl package. For complete documentation, see 
`get_dual_thimble_boundaries`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the dual thimbles.
- `params::Dict`: The parameters for the dual thimble calculation.
- `contributing::Bool`: Whether to include only contributing points.

# Returns
- `Vector{Saddle}`: A vector of the saddle points and their dual thimbles.
"""
function get_dual_thimble_boundaries(
    z::Union{Num, AbstractVector{Num}}, S::Num,
    params::Dict,
    domain::Vector{ComplexDomain},
    contributing::Bool
)::Vector{Saddle}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimble_boundaries(native_S, native_grad, native_hessian, params, domain, contributing)
end
