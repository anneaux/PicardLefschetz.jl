using Symbolics
using Base.Threads

export integrate_thimble
"""
    integrate_thimble(z, S, boundary, prefactor, params)

Integrates a Lefschetz thimble along its boundary using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `integrate_thimble`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `boundary::Any`: The boundary of the thimble to integrate along. (A set of flow points in 1D, or a mesh in 2D)
- `prefactor::Union{Num, AbstractVector{Num}}`: The symbolic prefactor of the integrand.
- `params::Dict`: The parameters for the integration.

# Returns
- `ComplexF64`: The integrated value.
"""
function integrate_thimble(
    z::Union{Num, AbstractVector{Num}}, S::Num,
    boundary::Any,
    prefactor::Union{Num, AbstractVector{Num}},
    params::Dict
)::Union{ComplexF64, AbstractVector}
    native_S = build_function(S, z, expression=Val{false})
    native_prefactor = prefactor isa AbstractArray ? build_function(prefactor, z, expression=Val{false})[1] : build_function(prefactor, z, expression=Val{false})

    return integrate_thimble(native_S, boundary, native_prefactor, params)
end

export integrate_thimble!
"""
    integrate_thimble!(z, S, saddle_point, prefactor, params)

Integrates a Lefschetz thimble for a given saddle point using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `integrate_thimble!`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The struct containing the saddle point and the precomputed thimble.
- `prefactor::Union{Num, AbstractVector{Num}}`: The symbolic prefactor of the integrand.
- `params::Dict`: The parameters for the integration.

# Returns
- `Nothing`
"""
function integrate_thimble!(
    z::Union{Num, AbstractVector{Num}}, S::Num,
    saddle_point::Types.Saddle,
    prefactor::Union{Num, AbstractVector{Num}}
)::Nothing
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})
    native_prefactor = prefactor isa AbstractArray ? build_function(prefactor, z, expression=Val{false})[1] : build_function(prefactor, z, expression=Val{false})

    return integrate_thimble!(native_S, native_grad, native_hessian, saddle_point, native_prefactor)
end

export integrate_thimbles
"""
    integrate_thimbles(z, S, domain, deformation_parameters, prefactor, params, mode)

Integrates the Lefschetz thimbles for a given domain using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `integrate_thimbles`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `domain::Vector{RealDomain}`: The domain over which to calculate the thimbles.
- `deformation_parameters::Vector{<:Number}`: A vector of initial point parameters.
- `prefactor::Union{Num, AbstractVector{Num}}`: The symbolic prefactor of the integrand.
- `params::Dict`: The parameters for the thimble calculation.
- `mode::String`: The mode of integration (`fixed` or `unfixed`).

# Returns
- `Tuple{Vector{ComplexF64},Int}`: A tuple containing the integrated values and the number of points evaluated.
"""
function integrate_thimbles(
    z::Union{Num, AbstractVector{Num}}, S::Num,
    domain::Vector{RealDomain},
    deformation_parameters::Vector{<:Number},
    prefactor::Union{Num, AbstractVector{Num}},
    params::Dict, mode::String
)::Union{Tuple{Vector{ComplexF64},Int}, Tuple{AbstractVector,Int}}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_prefactor = prefactor isa AbstractArray ? build_function(prefactor, z, expression=Val{false})[1] : build_function(prefactor, z, expression=Val{false})

    return integrate_thimbles(native_S, native_grad, domain, deformation_parameters, native_prefactor, params, mode)
end

export integrate_thimbles
"""
    integrate_thimbles(z, S, domain, params, prefactor; check)

Integrates the Lefschetz thimbles for all saddles using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `integrate_thimbles`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the thimbles.
- `params::Dict`: The parameters for the thimble calculation.
- `prefactor::Union{Num, AbstractVector{Num}}`: The symbolic prefactor of the integrand.
- `check::Function`: A function for checking equality of critical points.

# Returns
- `Vector{Types.Saddle}`: A vector of the saddle points, with their integrals populated.
"""
function integrate_thimbles(
    z::Union{Num, AbstractVector{Num}}, S::Num,
    domain::Vector{ComplexDomain},
    params::Dict, prefactor::Union{Num, AbstractVector{Num}};
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Types.Saddle}
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})
    native_prefactor = prefactor isa AbstractArray ? build_function(prefactor, z, expression=Val{false})[1] : build_function(prefactor, z, expression=Val{false})

    return integrate_thimbles(native_S, native_grad, native_hessian, domain, params, native_prefactor, check=check)
end
