using Symbolics

export find_analytic_saddles
"""
    find_analytic_saddles(z, derivative, initial_point, accuracy)

Finds the saddle points analytically using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. 
For complete documentation, see `find_analytic_saddles`.

# Arguments
- `z::Num`: The symbolic variable.
- `derivative::Num`: The symbolic first derivative of the action function.
- `initial_point::Vector{ComplexF64}`: The initial points to start the search from.
- `accuracy::Int64`: The accuracy (number of digits) to which the saddle points should be found.

# Returns
- A tuple or vector containing the found saddle points.
"""
function find_analytic_saddles(
    z::Num, derivative::Num,
    initial_point::Vector{ComplexF64},
    accuracy::Int64
)
    native_derivative = build_function(derivative, z, expression=Val{false})

    return find_analytic_saddles(native_derivative, initial_point, accuracy)
end

export find_numerical_saddles
"""
    find_numerical_saddles(z, derivative, domain, params; check)

Finds the saddle points numerically over a given domain using symbolic expressions. This 
is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `find_numerical_saddles`.

# Arguments
- `z::Num`: The symbolic variable.
- `derivative::Num`: The symbolic first derivative of the action function.
- `domain::Vector{ComplexDomain}`: The domain over which to search for the saddle points.
- `params::Dict`: The parameters for the numerical search.
- `check::Function`: A function used to check if two found saddle points are identical.

# Returns
- A vector of `Saddle` structs containing the found saddle points.
"""
function find_numerical_saddles(
    z::Num, derivative::Num,
    domain::Vector{ComplexDomain},
    params::Dict;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)
    native_derivative = build_function(derivative, z, expression=Val{false})

    return find_numerical_saddles(native_derivative, domain, params, check=check)
end

export check_contribution!
"""
    check_contribution!(z, S, saddle_point, domain, params; log_errors)

Checks whether a given saddle point contributes to the integral using symbolic expressions.
This is a wrapper function for compatibility with the Symbolics.jl package. For complete 
documentation, see `check_contribution!`.

# Arguments
- `z::Num`: The symbolic variable.
- `S::Num`: The symbolic action function.
- `saddle_point::Types.Saddle`: The saddle point struct to check.
- `domain::ComplexDomain`: The integration domain.
- `params::Dict`: Parameters for checking contribution.
- `log_errors::Bool`: Whether to log errors during the check.

# Returns
- `Nothing` (the `contributing` field of the `saddle_point` is modified in-place).
"""
function check_contribution!(
    z::Num, S::Num,
    saddle_point::Types.Saddle,
    domain::ComplexDomain,
    params::Dict;
    log_errors::Bool=false
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_derivative = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})

    check_contribution!(native_S, native_derivative, native_hessian, saddle_point, domain, params, log_errors=log_errors)
end
