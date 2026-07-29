using Symbolics

export solve_first_derivative
"""
    solve_first_derivative(z, derivative, initial_point, accuracy)

Finds the saddle points analytically using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. 
For complete documentation, see `solve_first_derivative`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `derivative::Union{Num, AbstractVector{Num}}`: The symbolic first derivative of the action function.
- `initial_point::Vector{ComplexF64}`: The initial points to start the search from.
- `accuracy::Int64`: The accuracy (number of digits) to which the saddle points should be found.

# Returns
- A tuple or vector containing the found saddle points.
"""
function solve_first_derivative(
    z::Union{Num,AbstractVector{Num}}, derivative::Union{Num,AbstractVector{Num}},
    initial_point::Vector{ComplexF64},
    accuracy::Int64
)::Vector{Types.Saddle}
    native_derivative = derivative isa AbstractArray ? build_function(derivative, z, expression=Val{false})[1] : build_function(derivative, z, expression=Val{false})
    return solve_first_derivative(native_derivative, initial_point, accuracy)
end

export find_saddles
"""
    find_saddles(z, derivative, domain, params; check)

Finds the saddle points numerically over a given domain using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `find_saddles`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `derivative::Union{Num, AbstractVector{Num}}`: The symbolic first derivative of the action function.
- `domain::Vector{ComplexDomain}`: The domain over which to search for the saddle points.
- `params::Dict`: The parameters for the numerical search.
- `check::Function`: A function used to check if two found saddle points are identical.

# Returns
- A vector of `Saddle` structs containing the found saddle points.
"""
function find_saddles(
    z::Union{Num,AbstractVector{Num}}, derivative::Union{Num,AbstractVector{Num}},
    domain::Vector{ComplexDomain},
    params::Dict;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Types.Saddle}
    native_derivative = derivative isa AbstractArray ? build_function(derivative, z, expression=Val{false})[1] : build_function(derivative, z, expression=Val{false})
    return find_saddles(native_derivative, domain, params, check=check)
end

export check_contribution!
"""
    check_contribution!(z, S, saddle_point, domain, params; log_errors)

Checks if a saddle point contributes to the integral over a given domain. This is a 
wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `check_contribution!`.

# Arguments
- `z::Union{Num, AbstractVector{Num}}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The struct containing the saddle point.
- `domain::ComplexDomain`: The domain over which to check the contribution.
- `params::Dict`: The parameters for the thimble boundary calculation.
- `log_errors::Bool`: Whether to log errors during the calculation.

# Returns
- `Nothing`
"""
function check_contribution!(
    z::Union{Num,AbstractVector{Num}}, S::Num,
    saddle_point::Types.Saddle,
    domain::Union{ComplexDomain,Vector{ComplexDomain}},
    params::Dict;
    log_errors::Bool=false
)::Nothing
    S_grad = z isa AbstractVector ? Symbolics.gradient(S, z) : Symbolics.gradient(S, [z])[1]
    S_hessian = z isa AbstractVector ? Symbolics.hessian(S, z) : Symbolics.hessian(S, [z])[1, 1]

    native_S = build_function(S, z, expression=Val{false})
    native_derivative = S_grad isa AbstractArray ? build_function(S_grad, z, expression=Val{false})[1] : build_function(S_grad, z, expression=Val{false})
    native_hessian = S_hessian isa AbstractArray ? build_function(S_hessian, z, expression=Val{false})[1] : build_function(S_hessian, z, expression=Val{false})

    check_contribution!(native_S, native_derivative, native_hessian, saddle_point, domain, params, log_errors=log_errors)
end

export get_intersection_number!
"""
    get_intersection_number!(z, S, saddle, params)

Calculates the intersection number between the thimble and the real axis using symbolic expressions.
This is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `get_intersection_number!`.

# Arguments
- `z::AbstractVector{Num}`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle::Types.Saddle`: The saddle point struct to check.
- `params::Dict`: Parameters for checking contribution.

# Returns
- `Nothing` (the `intersection_number` field of the `saddle` is modified in-place).
"""
function get_intersection_number!(z::AbstractVector{Num}, S::Num, saddle::Types.Saddle, params::Dict)::Nothing
    S_grad = Symbolics.gradient(S, z)[1]
    native_S = build_function(S, z, expression=Val{false})
    native_derivative = build_function(S_grad, z, expression=Val{false})

    return get_intersection_number!(native_S, native_derivative, saddle, params)
end
