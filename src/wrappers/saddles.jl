export find_analytic_saddles
function find_analytic_saddles(
    z::Num, derivative::Num,
    initial_point::Vector{ComplexF64},
    accuracy::Int64
)
    native_derivative = build_function(derivative, z, expression=Val{false})

    return find_analytic_saddles(native_derivative, initial_point, accuracy)
end

export find_numerical_saddles
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
function check_contribution!(
    z::Num, S::Num,
    saddle_point::Types.Saddle,
    domain::ComplexDomain,
    params::Dict;
    log_errors::Bool=false
)
    S_grad = Symbolics.gradient(S, z)
    S_hessian = Symbolics.hessian(S, z)
    native_S = build_function(S, z, expression=Val{false})
    native_derivative = build_function(S_grad, z, expression=Val{false})[1]
    native_hessian = build_function(S_hessian, z, expression=Val{false})[1]

    check_contribution!(native_S, native_derivative, native_hessian, saddle_point, domain, params, log_errors=log_errors)
end
