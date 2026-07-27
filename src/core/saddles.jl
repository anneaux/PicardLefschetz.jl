module Saddle

using ..Methods1D
using ..Methods2D
using ..Types

export find_analytic_saddles
"""
    find_analytic_saddles(derivative, initial_point, accuracy)

Finds the saddle points analytically using the provided initial points. This function solves for the zeros of the first derivative in the analytic continuation.

# Arguments
- `derivative::Function`: The first derivative of the action function.
- `initial_point::Vector{ComplexF64}`: The initial points to start the search from.
- `accuracy::Int64`: The accuracy (number of digits) to which the saddle points should be found.

# Returns
- A tuple or vector containing the found saddle points.
"""
function find_analytic_saddles(derivative::Function, initial_point::Vector{ComplexF64}, accuracy::Int64)
    if length(initial_point) == 2
        # 2D case
        return Methods2D.SaddlePoint.solve_first_drv(derivative, initial_point, digits=accuracy)
    elseif length(initial_point) == 1
        # 1D case
        tmp = [real(initial_point[1]), imag(initial_point[1])]
        return Methods1D.CriticalPoints.solve_first_derivative(derivative, tmp, accuracy)
    else
        throw(error("Input vector unsupported length."))
    end
end

export find_numerical_saddles
"""
    find_numerical_saddles(derivative, domain, params; check)

Finds the saddle points numerically over a given domain using a Sobol sequence for initial points. The
parameters for this function are listed below:

| Parameter | Required | Type | Description |
| --------- | -------- | ---- | ----------- |
| `point_count` | Yes | `Int` | The initial number of points in the Sobol sequence. |
| `accuracy` | Yes | `Int` | The accuracy (number of digits) to which the saddle points should be found. |

# Arguments
- `derivative::Function`: The first derivative of the action function.
- `domain::Vector{ComplexDomain}`: The domain over which to search for the saddle points.
- `params::Dict`: The parameters for the numerical search, including `point_count` and `accuracy`.
- `check::Function`: A function used to check if two found saddle points are identical (default `!isequal`).

# Returns
- A vector of `Saddle` structs containing the found saddle points.
"""
function find_numerical_saddles(
    derivative::Function,
    domain::Vector{ComplexDomain},
    params::Dict;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Saddle}

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

| Parameter | Required | Type | Description |
| --------- | -------- | ---- | ----------- |
| `grid_resolution` | Yes | `Int` | The number of points to use for discretizing the thimble paths. |
| `flow_step_factor` | No | `Float64` | The step size factor for the flow equation. (This parameter is only required in 2D.)|
| `initial_necklace_size` | No | `Int` | The initial number of points in the dual Lefschetz thimble contour. (This parameter is only required in 2D.)|
| `max_iterations` | No | `Int` | The maximum number of iterations for the flow equation. (This parameter is only required in 2D.)|
| `init_pertubation_radius` | No | `Float64` | The initial radius of the perturbation used to generate the necklace. (This parameter is only required in 2D.)|
| `subdivision_threshold` | No | `Float64` | The threshold for subdividing the necklace to improve accuracy. (This parameter is only required in 2D.)|

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The first derivative (gradient) of the action function.
- `S_hessian::Function`: The second derivative (Hessian) of the action function.
- `saddle_point::Types.Saddle`: The saddle point struct to check.
- `domain::ComplexDomain`: The integration domain.
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
    domain::ComplexDomain,
    params::Dict;
    log_errors::Bool=false
)::Nothing
    grid_resolution = params["grid_resolution"]

    contributing = if length(saddle_point.saddle) == 2
        flow_step_factor = params["flow_step_factor"]
        initial_necklace_size = params["initial_necklace_size"]
        max_iterations = params["max_iterations"]
        init_pertubation_radius = params["init_pertubation_radius"]
        subdivision_threshold = params["subdividethreshold"]

        Methods2D.SaddlePoint.check_contribution(
            S, S_grad, S_hessian, saddle_point,
            Ntimes=grid_resolution, logerrors=log_errors,
            flowstepfactor=flow_step_factor, initial_necklace_size=initial_necklace_size,
            max_iterations=max_iterations, init_pertubation_radius=init_pertubation_radius,
            subdivision_threshold=subdivision_threshold
        )
    elseif length(saddle_point.saddle) == 1
        Methods1D.SaddlePoint.is_contributing(saddle_point, S, domain.min, domain.max, Ntimes=grid_resolution)
    end

    saddle_point.contributing = contributing

end

include("../wrappers/saddles.jl")

end