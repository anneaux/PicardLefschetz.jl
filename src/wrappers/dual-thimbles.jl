using Symbolics

export get_dual_thimble!
function get_dual_thimble!(
    z::Num, S::Num,
    saddle_point::Saddle,
    params::Dict
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimble!(native_S, native_grad, native_hessian, saddle_point, params)
end

export get_dual_thimbles
function get_dual_thimbles(
    z::Num, S::Num, params::Dict,
    domain::Vector{ComplexDomain},
    contributing::Bool
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimbles(native_S, native_grad, native_hessian, params, domain, contributing)
end

export get_dual_thimble_boundary!
function get_dual_thimble_boundary!(
    z::Num, S::Num,
    saddle_point::Saddle,
    params::Dict
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimble_boundary!(native_S, native_grad, native_hessian, saddle_point, params)
end

export get_dual_thimble_boundaries
function get_dual_thimble_boundaries(
    z::Num, S::Num,
    params::Dict,
    domain::Vector{ComplexDomain},
    contributing::Bool
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})

    return get_dual_thimble_boundaries(native_S, native_grad, native_hessian, params, domain, contributing)
end
