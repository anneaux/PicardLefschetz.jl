using Symbolics

export integrate_thimble
function integrate_thimble(
    z::Num, S::Num,
    boundary::Any,
    prefactor::Num,
    params::Dict
)
    native_S = build_function(S, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimble(native_S, boundary, native_prefactor, params)
end

export integrate_thimble!
function integrate_thimble!(
    z::Num, S::Num,
    saddle_point::Types.Saddle,
    prefactor::Num
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimble!(native_S, native_grad, native_hessian, saddle_point, native_prefactor)
end

export integrate_thimbles
function integrate_thimbles(
    z::Num, S::Num,
    domain::Vector{RealDomain},
    deformation_parameters::Vector{<:Number},
    prefactor::Num,
    params::Dict, mode::String
)
    S_grad = Symbolics.gradient(S, [z])[1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimbles(native_S, native_grad, domain, deformation_parameters, native_prefactor, params, mode)
end

export integrate_thimbles
function integrate_thimbles(
    z::Num, S::Num,
    domain::Vector{ComplexDomain},
    params::Dict, prefactor::Num;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)
    S_grad = Symbolics.gradient(S, [z])[1]
    S_hessian = Symbolics.hessian(S, [z])[1, 1]
    native_S = build_function(S, z, expression=Val{false})
    native_grad = build_function(S_grad, z, expression=Val{false})
    native_hessian = build_function(S_hessian, z, expression=Val{false})
    native_prefactor = build_function(prefactor, z, expression=Val{false})

    return integrate_thimbles(native_S, native_grad, native_hessian, domain, params, native_prefactor, check=check)
end
