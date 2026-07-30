module DualThimble

using ..Methods1D
using ..Methods2D
using ..Types
using ..Saddle: find_saddles

# Gets the dual thimble for a given saddle point.
export get_dual_thimble!
"""
    get_dual_thimble!(S, S_grad, S_hessian, saddle_point, params)

Calculates the dual thimble, using gradient ascent to be able to calculate the dual thimble. Gradient ascent is gradient descent in reverse, which is done
by solving the ODE along a time parameter \$ \\tau \$ to find the dual thimble. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = \\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Required | Type | Description                                                                                                                                   |
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `flow_step_factor`    | Yes      | `Float64` | The step size of the gradient ascent solver.                                                                                                  |
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 |
| `height_threshold`    | Yes      | `Float64` | The maximal magnitude of the imaginary component during gradient ascent. This is used when checking if the saddle point contributes.        |
| `init_point_count`    | No       | `Int` | The initial number of points for the dual Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) |
| `init_perturbation_radius` | No  | `Float64` | The initial radius for the dual Lefschetz thimble contour. This is the contour evolved to give the dual thimble. (This parameter is only required when the dimension of the saddle is 2.) |
| `subdivision_threshold` | No     | `Float64` | The threshold for subdivding the segments between the points of the dual Lefschetz thimble. (This parameter is only required when the dimension of the saddle is 2.) |


# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `saddle_point::Saddle`: The struct containing the saddle point and the related functions.
- `params::Dict`: The parameters for the dual thimble calculation. All parameters listed above are to be included in this dictionary.

# Returns
- `Nothing`
"""
function get_dual_thimble!(S::Function, S_grad::Function, S_hessian::Function, saddle_point::Saddle, params::Dict)::Nothing
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

        necklace, quadrangles, points = Methods2D.DualThimble.get_necklace(
            S, S_grad, S_hessian,
            saddle_point,
            Ninit=init_point_count,
            Ncounter=max_iterations,
            eigvecfactorinit=init_perturbation_radius,
            flowstepfactor=flow_step_factor,
            subdividethreshold=subdivision_threshold
        )
        saddle_point.dual_thimble = (points, quadrangles)
    end

    return nothing
end

# Gets the dual thimbles for all saddle points.
export get_dual_thimbles
"""
    get_dual_thimbles(S, S_grad, S_hessian, params, domain, contributing)

Calculates the dual thimble, using gradient ascent to be able to calculate the dual thimble. Gradient ascent is gradient descent in reverse, which is done
by solving the ODE along a time parameter \$ \\tau \$ to find the dual thimble. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = \\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Required | Type | Description                                                                                                                                   |
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `flow_step_factor`    | Yes      | `Float64` | The step size of the gradient ascent solver.                                                                                                  |
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 |
| `height_threshold`    | Yes      | `Float64` | The maximal magnitude of the imaginary component during gradient ascent. This is used when checking if the saddle point contributes.        |
| `init_point_count`    | No       | `Int` | The initial number of points for the dual Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) |
| `init_perturbation_radius` | No  | `Float64` | The initial radius for the dual Lefschetz thimble contour. This is the contour evolved to give the dual thimble. (This parameter is only required when the dimension of the saddle is 2.) |
| `subdivision_threshold` | No     | `Float64` | The threshold for subdivding the segments between the points of the dual Lefschetz thimble. (This parameter is only required when the dimension of the saddle is 2.) |


# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `params::Dict`: The parameters for the dual thimble calculation. All parameters listed above are to be included in this dictionary.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the dual thimbles. In the 1D case, the domain should be only one `ComplexDomain`, and in 2D, it should be `ComplexDomain`s.
- `contributing::Bool`: Whether to include only contributing points. 

# Returns
- `Vector{Saddle}`: A vector of the saddle points and their dual thimbles.
"""
function get_dual_thimbles(S::Function, S_grad::Function, S_hessian::Function, params::Dict, domain::Vector{ComplexDomain}, contributing::Bool)::Vector{Saddle}
    saddles = find_saddles(S_grad, domain, params)
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
"""
    get_dual_thimble_boundary!(S, S_grad, S_hessian, saddle_point, params)

Calculates the boundary of the dual thimble, using gradient ascent to be able to calculate the dual thimble boundary. Gradient ascent is gradient descent in reverse, which is done
by solving the ODE along a time parameter \$ \\tau \$ to find the dual thimble boundary. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = \\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Required | Type | Description                                                                                                                                   |
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `flow_step_factor`    | Yes      | `Float64` | The step size of the gradient ascent solver.                                                                                                  |
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 |
| `height_threshold`    | Yes      | `Float64` | The maximal magnitude of the imaginary component during gradient ascent. This is used when checking if the saddle point contributes.        |
| `init_point_count`    | No       | `Int` | The initial number of points for the dual Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) |
| `init_perturbation_radius` | No  | `Float64` | The initial radius for the dual Lefschetz thimble contour. This is the contour evolved to give the dual thimble. (This parameter is only required when the dimension of the saddle is 2.) |
| `subdivision_threshold` | No     | `Float64` | The threshold for subdivding the segments between the points of the dual Lefschetz thimble. (This parameter is only required when the dimension of the saddle is 2.) |


# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `saddle_point::Saddle`: The struct containing the saddle point and the related functions.
- `params::Dict`: The parameters for the dual thimble calculation. All parameters listed above are to be included in this dictionary.

# Returns
- `Nothing`
"""
function get_dual_thimble_boundary!(S::Function, S_grad::Function, S_hessian::Function, saddle_point::Types.Saddle, params::Dict)::Nothing
    if length(saddle_point) == 1
        saddle = saddle_point.saddle
        flow_step_factor = params["flow_step_factor"]
        max_iterations = params["max_iterations"]
        height_threshold = params["height_threshold"]
        thimbles, contributing = Methods1D.PathFlow.flow_up(S, S_grad, saddle, flow_step_factor, height_threshold, max_iterations)
        saddle_point.dual_thimble_boundary = thimbles
    elseif length(saddle_point) == 2
        flow_step_factor = params["flow_step_factor"]
        max_iterations = params["max_iterations"]
        init_point_count = params["init_point_count"]
        init_perturbation_radius = params["init_perturbation_radius"]
        subdivision_threshold = params["subdivision_threshold"]
        necklace, quadrangles, points = Methods2D.DualThimble.get_necklace_solver(
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
"""
    get_dual_thimble_boundaries(S, S_grad, S_hessian, params, domain, contributing)

Calculates the boundary of the dual thimbles for all saddles, using gradient ascent to be able to calculate the dual thimble boundary. Gradient ascent is gradient descent in reverse, which is done
by solving the ODE along a time parameter \$ \\tau \$ to find the dual thimble boundary. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = \\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Required | Type | Description                                                                                                                                   |
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `flow_step_factor`    | Yes      | `Float64` | The step size of the gradient ascent solver.                                                                                                  |
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 |
| `height_threshold`    | Yes      | `Float64` | The maximal magnitude of the imaginary component during gradient ascent. This is used when checking if the saddle point contributes.        |
| `init_point_count`    | No       | `Int` | The initial number of points for the dual Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) |
| `init_perturbation_radius` | No  | `Float64` | The initial radius for the dual Lefschetz thimble contour. This is the contour evolved to give the dual thimble. (This parameter is only required when the dimension of the saddle is 2.) |
| `subdivision_threshold` | No     | `Float64` | The threshold for subdivding the segments between the points of the dual Lefschetz thimble. (This parameter is only required when the dimension of the saddle is 2.) |


# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `params::Dict`: The parameters for the dual thimble calculation. All parameters listed above are to be included in this dictionary.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the dual thimbles. In the 1D case, the domain should be only one `ComplexDomain`, and in 2D, it should be `ComplexDomain`s.
- `contributing::Bool`: Whether to include only contributing points. 

# Returns
- `Vector{Saddle}`: A vector of the saddle points and their dual thimbles.
"""
function get_dual_thimble_boundaries(S::Function, S_grad::Function, S_hessian::Function, params::Dict, domain::Vector{ComplexDomain}, contributing::Bool)::Vector{Saddle}
    saddles = find_saddles(S_grad, domain, params)
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