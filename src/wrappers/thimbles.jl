using Symbolics

export get_thimble!
function get_thimble!(
    z::Num, S::Num,
    saddle_point::Types.Saddle,
    params::Dict;
    mesh_type::String
)
    S_grad = Symbolics.gradient(S, z)
    S_hessian = Symbolics.hessian(S, z)
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})[1]
    native_hessian = build_function(S_hessian, z, expression=Val{false})[1]

    return get_thimble!(native_S, native_grad, native_hessian, saddle_point, params, mesh_type=mesh_type)
end

export get_thimbles
function get_thimbles(
    z::Num, S::Num,
    params::Dict,
    domain::Vector{RealDomain},
    contributing::Bool;
    mesh_type::String
)
    S_grad = Symbolics.gradient(S, z)
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})[1]

    return get_thimbles(native_S, native_grad, params, domain, contributing, mesh_type=mesh_type)
end

export get_thimble_boundary!
function get_thimble_boundary!(
    z::Num, S::Num,
    saddle_point::Types.Saddle,
    params::Dict;
    mesh_type::String
)
    S_grad = Symbolics.gradient(S, z)
    S_hessian = Symbolics.hessian(S, z)
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})[1]
    native_hessian = build_function(S_hessian, z, expression=Val{false})[1]

    return get_thimble_boundary!(native_S, native_grad, native_hessian, saddle_point, params, mesh_type=mesh_type)
end

export get_thimble_boundaries
function get_thimble_boundaries(
    z::Num, S::Num,
    domain::Vector{ComplexDomain},
    params::Dict, contributing::Bool;
    mesh_type::String,
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)
    S_grad = Symbolics.gradient(S, z)
    S_hessian = Symbolics.hessian(S, z)
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})[1]
    native_hessian = build_function(S_hessian, z, expression=Val{false})[1]

    return get_thimble_boundaries(native_S, native_grad, native_hessian, domain, params, contributing, mesh_type=mesh_type, check=check)
end
