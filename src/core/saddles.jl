module Saddle

using ..Methods1D
using ..Methods2D
using ..Types

export solve_first_derivative
"""
    solve_first_derivative(derivative, initial_point, accuracy)

Finds the saddle point using the provided initial point. This function solves for the zeros of the first derivative in the analytic continuation, by using the Newton-Raphson method.

# Arguments
- `derivative::Function`: The first derivative of the action function.
- `initial_point::Vector{ComplexF64}`: The initial point to start the search from.
- `accuracy::Int64`: The accuracy (number of digits) to which the saddle point should be found.

# Returns
- `Saddle`: A saddle containing the found saddle point.
"""
function solve_first_derivative(derivative::Function, initial_point::Vector{ComplexF64}, accuracy::Int64)::Types.Saddle
    if length(initial_point) == 2
        # 2D case
        t_1, t_2 = Methods2D.SaddlePoint.solve_first_drv(derivative, initial_point, digits=accuracy)
        return Saddle([t_1, t_2])
    elseif length(initial_point) == 1
        # 1D case
        tmp = [real(initial_point[1]), imag(initial_point[1])]
        return Methods1D.CriticalPoints.solve_first_derivative(derivative, tmp, accuracy)
    else
        throw(error("Input vector unsupported length."))
    end
end

export find_saddles
"""
    find_saddles(derivative, domain, params; check)

Finds the saddle points numerically over a given domain using a Sobol sequence for initial points, and flowing them using gradient descent. The
parameters for this function are listed below:

| Parameter | Always Required | Type | Description | Heuristic |
| --------- | -------- | ---- | ----------- | ------ |
| `point_count` | Yes | `Int` | The initial number of points in the Sobol sequence. | No |
| `accuracy` | Yes | `Int` | The accuracy (number of digits) to which the saddle points should be found. | No |

# Arguments
- `derivative::Function`: The first derivative of the action function.
- `domain::Vector{ComplexDomain}`: The domain over which to search for the saddle points.
- `params::Dict`: The parameters for the numerical search, including `point_count` and `accuracy`.
- `check::Function`: A function used to check if two found saddle points are identical (default `!isequal`).

# Returns
- A vector of `Saddle` structs containing the found saddle points.
"""
function find_saddles(
    derivative::Function,
    domain::Vector{ComplexDomain},
    params::Dict;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Types.Saddle}

    point_count = params["point_count"]
    accuracy = params["accuracy"]

    if length(domain) == 2
        # 2D case
        return Methods2D.SaddlePoint.find_saddles_sobol(derivative, domain[1], domain[2], N=point_count, digits=accuracy, check=check)
    elseif length(domain) == 1
        # 1D case 
        return Methods1D.CriticalPoints.find_saddles_sobol(derivative, domain[1].min, domain[1].max, point_count)
    else
        throw(error("Input vector unsupported length."))
    end
end

export check_contribution!
"""
    check_contribution!(S, S_grad, S_hessian, saddle_point, domain, params; log_errors)

Checks whether a given saddle point contributes to the integral, using gradient ascent to check whether the thimble contributes. 
The parameters for this function are listed below:

| Parameter | Always Required | Type | Description | Heuristic |
| --------- | -------- | ---- | ----------- | ------ |
| `flow_step_factor` | No | `Real` | The step size factor for the flow equation. (This parameter is only required in 2D.)| Yes |
| `init_point_count` | No | `Int` | The initial number of points in the dual Lefschetz thimble contour. (This parameter is only required in 2D.)| Yes |
| `max_iterations` | No | `Int` | The maximum number of iterations for the flow equation. (This parameter is only required in 2D.)| Yes |
| `init_perturbation_radius` | No | `Real` | The initial radius of the perturbation used to generate the necklace. (This parameter is only required in 2D.)| Yes |
| `subdivision_threshold` | No | `Real` | The threshold for subdividing the necklace to improve accuracy. (This parameter is only required in 2D.)| Yes |

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The first derivative (gradient) of the action function.
- `S_hessian::Function`: The second derivative (Hessian) of the action function.
- `saddle_point::Types.Saddle`: The saddle point struct to check.
- `check::Function`: A function used to check if the saddle point is within the domain. Should accept the coordinates in a vector of complex numbers.
- `params::Dict`: Parameters for checking contribution.
- `log_errors::Bool`: Whether to log errors during the check (default `false`).

# Returns
- `Nothing` (the `contributing` field of the `saddle_point` is modified in-place).
"""
function check_contribution!(
    S::Function,
    S_grad::Function,
    S_hessian::Function,
    saddle_point::Types.Saddle,
    check::Function,
    params::Dict;
    log_errors::Bool=false
)::Nothing
    contributing = if length(saddle_point.saddle) == 2
        flow_step_factor = Float64(params["flow_step_factor"])
        initial_necklace_size = params["init_point_count"]
        max_iterations = params["max_iterations"]
        init_perturbation_radius = Float64(params["init_perturbation_radius"])
        subdivision_threshold = Float64(params["subdivision_threshold"])

        Methods2D.SaddlePoint.check_contribution(
            S, S_grad, S_hessian, saddle_point, check,
            logerrors=log_errors, flowstepfactor=flow_step_factor,
            initial_necklace_size=initial_necklace_size, max_iterations=max_iterations,
            init_perturbation_radius=init_perturbation_radius,
            subdivision_threshold=subdivision_threshold
        )
    elseif length(saddle_point.saddle) == 1
        height_threshold = Float64(params["height_threshold"])
        dual_thimble, contributing = Methods1D.PathFlow.flow_up(S, S_grad, saddle_point, flow_step_factor, height_threshold, max_iterations)
        if abs(imag(saddle_point.coords[1])) > 1e-10
            return true
        end
        intersection_point = Methods1D.PathFlow.find_intersection_point(dual_thimble)
        if contributing && check([intersection_point.coords[1]])
            return true
        end
        false
    end

    saddle_point.contributing = contributing
    return nothing
end

export get_intersection_number!
"""
    get_intersection_number(S, S_grad, S_hessian, saddle, params)

Calculates the intersection number of the dual Lefschetz thimble with the real integral domain. 
The parameters for this function are listed below:

| Parameter | Always Required | Type | Description | Heuristic |
| --------- | -------- | ---- | ----------- | ------ |
| `flow_step_factor` | Yes | `Real` | The step size factor for the flow equation. | Yes |
| `max_iterations` | Yes | `Int` | The maximum number of iterations for the flow equation. | Yes |
| `init_perturbation_radius` | Yes | `Real` | The initial radius of the perturbation used to generate the necklace. | Yes |
| `subdivision_threshold` | Yes | `Real` | The threshold for subdividing the necklace to improve accuracy. | Yes |
| `height_threshold` | Yes | `Real` | The threshold for the cutoff of the thimble/dual thimble, in the imaginary magnitude of the action. | Yes |
| `gradient_normalisation_threshold` | Yes | `Real` | The threshold for normalising the gradient during gradient flow. | Yes |

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The first derivative (gradient) of the action function.
- `saddle_point::Types.Saddle`: The saddle point struct to check.
- `params::Dict`: Parameters for checking contribution.

# Returns
- `Nothing` (the `intersection_number` field of the `saddle` is modified in-place).
"""
function get_intersection_number!(S::Function, S_grad::Function, saddle::Types.Saddle, params::Dict)::Nothing
    if length(saddle.saddle) == 1
        return Methods1D.SaddlePoint.get_intersection_number!(S, S_grad, saddle, params)
    elseif length(saddle.saddle) == 2
        return Methods2D.SaddlePoint.get_intersection_number!(S, saddle, params)
    end
end

include("../wrappers/saddles.jl")

end