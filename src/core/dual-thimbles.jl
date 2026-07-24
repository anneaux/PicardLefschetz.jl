module DualThimble

using ..Methods1D
using ..Methods2D
using ..Types
using ..Saddle: find_numerical_saddles

# Gets the dual thimble for a given saddle point.
export get_dual_thimble!
function get_dual_thimble!(S::Function, S_grad::Function, S_hessian::Function, saddle_point::Saddle, params::Dict)
    flow_step_factor = params["flow_step_factor"]
    max_iterations = params["max_iterations"]
    height_threshold = params["height_threshold"]
    if length(saddle_point) == 1
        thimbles, contributing = Methods1D.PathFlow.flow_up(S, S_grad, saddle_point.saddle, flow_step_factor, height_threshold, max_iterations)
        saddle_point.dual_thimble = thimbles
    elseif length(saddle_point) == 2
        init_point_count = params["init_point_count"]
        init_perturbation_radius = params["init_perturbation_radius"]
        subdivision_threshold = params["subdivision_threshold"]

        necklace, quadrangles = Methods2D.DualThimble.get_necklace(
            S, S_grad, S_hessian,
            saddle_point,
            Ninit=init_point_count,
            Ncounter=max_iterations,
            eigvecfactorinit=init_perturbation_radius,
            flowstepfactor=flow_step_factor,
            subdividethreshold=subdivision_threshold
        )
        saddle_point.dual_thimble = quadrangles
    end

    return nothing
end

# Gets the dual thimbles for all saddle points.
export get_dual_thimbles
function get_dual_thimbles(S::Function, S_grad::Function, S_hessian::Function, params::Dict, domain::Vector{ComplexDomain}, contributing::Bool)
    saddles = find_numerical_saddles(S_grad, domain, params)
    for saddle in saddles
        if contributing
            check_contribution!(S, S_grad, S_hessian, saddle, domain[1], params)
            if saddle.contributing
                get_dual_thimble!(S, S_grad, S_hessian, saddle, params)
            end
        else
            get_dual_thimble!(S, S_grad, S_hessian, saddle, params)
        end
    end

    return saddles
end

# Gets the dual thimble boundary for a given saddle point.
export get_dual_thimble_boundary!
function get_dual_thimble_boundary!(S::Function, S_grad::Function, S_hessian::Function, saddle_point::Types.Saddle, params::Dict)
    if length(saddle_point) == 1
        saddle = saddle_point.saddle
        flow_step_factor = params["flow_step_factor"]
        max_iterations = params["max_iterations"]
        height_threshold = params["height_threshold"]
        thimbles, contributing = Methods1D.PathFlow.flow_up(S, S_grad, saddle, flow_step_factor, height_threshold, max_iterations)
        saddle_point.dual_thimble_boundary = thimbles
    elseif length(saddle_point) == 2
        init_point_count = params["init_point_count"]
        init_perturbation_radius = params["init_perturbation_radius"]
        subdivision_threshold = params["subdivision_threshold"]
        necklace, quadrangles = Methods2D.DualThimble.get_necklace_solver(
            S, S_grad, S_hessian,
            saddle_point[1], saddle_point[2],
            Ninit=init_point_count,
            Ncounter=max_iterations,
            eigvecfactorinit=init_perturbation_radius,
            flowstepfactor=flow_step_factor,
            subdividethreshold=subdivision_threshold
        )

        saddle_point.dual_thimble_boundary = necklace
    end

    return nothing
end

# Gets the dual thimble boundary for all saddle points.
export get_dual_thimble_boundaries
function get_dual_thimble_boundaries(S::Function, S_grad::Function, S_hessian::Function, params::Dict, domain::Vector{ComplexDomain}, contributing::Bool)
    saddles = find_numerical_saddles(S_grad, domain, params)
    for saddle in saddles
        if contributing
            check_contribution!(S, S_grad, S_hessian, saddle, domain[1], params)
            if saddle.contributing
                get_dual_thimble_boundary!(S, S_grad, S_hessian, saddle, params)
            end
        else
            get_dual_thimble_boundary!(S, S_grad, S_hessian, saddle, params)
        end
    end
    return saddles
end

include("../wrappers/dual-thimbles.jl")

end