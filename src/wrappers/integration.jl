using Symbolics

export integrate_thimble
"""
    integrate_thimble(z, S, boundary, prefactor, params)

Integrates the given symbolic action function over a precomputed thimble boundary. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `integrate_thimble`.

# Arguments
- `z::Num`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `boundary::Any`: The precomputed boundary over which to integrate.
- `prefactor::Num`: A symbolic expression for the prefactor.
- `params::Dict`: Integration parameters.

# Returns
- `Vector{ComplexF64}`: The result of the integration.
"""
function integrate_thimble(
    z::Num, S::Num,
    boundary::Any,
    prefactor::Num,
    params::Dict
)::Vector{ComplexF64}
    native_S = build_function(S, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimble(native_S, boundary, native_prefactor, params)
end

export integrate_thimble!
"""
    integrate_thimble!(z, S, saddle_point, prefactor)

Integrates the symbolic action function around a single saddle point without a precomputed boundary. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete documentation, see 
`integrate_thimble!`.

# Arguments
- `z::Num`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The saddle point struct, updated with the `integral` result.
- `prefactor::Num`: A symbolic expression for the prefactor.

# Returns
- `Nothing` (the `integral` field of the `saddle_point` is modified in-place).
"""
function integrate_thimble!(
    z::Num, S::Num,
    saddle_point::Types.Saddle,
    prefactor::Num
)::Nothing
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimble!(native_S, native_grad, native_hessian, saddle_point, native_prefactor)
end

export integrate_thimbles
"""
    integrate_thimbles(z, S, domain, deformation_parameters, prefactor, params, mode)

Integrates the symbolic action function over the specified real domain by deforming the contour along the steepest descent paths.
This is a wrapper function for compatibility with the Symbolics.jl package. For complete documentation, see 
`integrate_thimbles`.

# Arguments
- `z::Num`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `domain::Vector{RealDomain}`: The initial integration domain.
- `deformation_parameters::Vector{<:Number}`: The parameters used to deform the integration contour.
- `prefactor::Num`: A symbolic expression for the prefactor.
- `params::Dict`: Integration and flow parameters.
- `mode::String`: The integration mode.

# Returns
- `Tuple{Vector{ComplexF64}, Int}`: A tuple of the integral value, and the number of simplices used to evaluate the integral.
"""
function integrate_thimbles(
    z::Num, S::Num,
    domain::Vector{RealDomain},
    deformation_parameters::Vector{<:Number},
    prefactor::Num,
    params::Dict, mode::String
)::Tuple{Vector{ComplexF64},Int}
    S_grad = Symbolics.gradient(S, [z])[1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimbles(native_S, native_grad, domain, deformation_parameters, native_prefactor, params, mode)
end

export integrate_thimbles
"""
    integrate_thimbles(z, S, domain, params, prefactor; check)

Integrates around all contributing thimbles using the saddle point approximation method with symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete documentation, see 
`integrate_thimbles`.

# Arguments
- `z::Num`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `domain::Vector{ComplexDomain}`: The domain over which to search for saddle points and integrate.
- `params::Dict`: Parameters for saddle point finding and integration.
- `prefactor::Num`: A symbolic expression for the prefactor.
- `check::Function`: A function used to check if two found saddle points are identical.

# Returns
- `Vector{Types.Saddle}`: The saddle points, with their computed Saddle Point Method approximation integrals.
"""
function integrate_thimbles(
    z::Num, S::Num,
    domain::Vector{ComplexDomain},
    params::Dict, prefactor::Num;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Types.Saddle}
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimbles(native_S, native_grad, native_hessian, domain, params, native_prefactor, check=check)
end
