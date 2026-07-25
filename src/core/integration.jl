module Integration

using ..Methods1D
using ..Methods2D
using ..Types
using ..Saddle

# Integrate using a given boundary around one thimble, i.e. boundary can be precomputed in this case.
export integrate_thimble
"""
    integrate_thimble(S, boundary, prefactor, params)

Integrates the given action and prefactor functions over a precomputed thimble boundary. The algorithm used is the Gauss-Legendre quadrature method on each simplex of the boundary.
This package is responsible for solving integrals of the form
\$\$
I = \\int^a_b f(z)e^{iS(z)}dz
\$\$

where S(z) is the action function which faster oscillation, and f(z) is the prefactor function with small oscillation. The parameters for this function are listed as such:
| Parameter             | Required | Type | Description                                                                                                                                   |
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `GL_order`    | No      | `Int` | The order of the Gauss-Legendre quadrature. (This parameter is only required if the dimension is 2, and the simplex is a quadrangle). |
| `simplex_order`      | No      | `Int` | The order of the simplex. (This parameter is only required if the dimension is 2, and the simplex is a triangle). |
| `output_dim`    | No      | `Int` | The output dimension of the integral. (This parameter is only required if the dimension is 2, and the simplex is a triangle). |

# Arguments
- `S::Function`: The action function.
- `boundary::Any`: The precomputed boundary over which to integrate. This can be a tuple for 1D or a vector of simplices (quadrilaterals or triangles) for 2D.
- `prefactor::Function`: The prefactor function for the integrand, f(z).
- `params::Dict`: Integration parameters

# Returns
- `Any`: Result of the integration.
"""
function integrate_thimble(S::Function, boundary::Any, prefactor::Function, params::Dict)
    if typeof(boundary) <: Tuple && length(boundary) == 2
        # 1D case
        return Methods1D.Integration.integrate_thimble(S, boundary[1], boundary[2])
    elseif boundary isa Vector{Simplex{4,FlowPoint}}
        # 2D case
        integral = 0
        for quad in boundary
            integral += Methods2D.Quadrilateral.Integration.integrate_quadrilateral(S, quad, params["GL_order"], prefactor=prefactor)
        end

        return integral
    elseif boundary isa Vector{Simplex{3,FlowPoint}}
        integral = 0
        for triangle in boundary
            integral += Methods2D.Triangle.Integration.integrate_triangle(S, triangle, prefactor=prefactor, order=params["simplex_order"], dim=params["output_dim"])
        end

        return integral
    end
end

# Integrate without a given boundary around one thimble, essentially steepest descent until integral converges to the required precision, or the number of flow steps is exceeded.
export integrate_thimble!
"""
    integrate_thimble!(S, S_grad, S_hessian, saddle_point, prefactor)

Integrates the action function around a single saddle point by using steepest descent without a precomputed boundary, continuing until the integral converges to the required precision 
or the number of flow steps is exceeded. This package is responsible for solving integrals of the form
\$\$
I = \\int^a_b f(z)e^{iS(z)}dz
\$\$

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The first derivative (gradient) of the action function.
- `S_hessian::Function`: The second derivative (Hessian) of the action function.
- `saddle_point::Types.Saddle`: The saddle point struct, which will be updated with the `integral` result.
- `prefactor::Function`: A function that provides the prefactor for the integrand.

# Returns
- `Nothing` (the `integral` field of the `saddle_point` is modified in-place).
"""
function integrate_thimble!(S::Function, S_grad::Function, S_hessian::Function, saddle_point::Types.Saddle, prefactor::Function)
    if length(saddle_point.saddle) == 1
        # 1D case
        integral = Methods1D.SaddlePoint.integrate_around_saddle_point(saddle_point, S, S_grad, S_hessian, prefactor=prefactor)
    elseif length(saddle_point.saddle) == 2
        # 2D case
        integral = Methods2D.SaddlePoint.saddles_gaussian_contribution(S, S_hessian, saddle_point, prefactor=prefactor)
    end

    saddle_point.integral = integral
    return nothing
end

# Integrate without a given boundary around all contributing thimbles, essentially using steepest descent until integral converges to the required precision, or the number of flow steps is exceeded.
export integrate_thimbles
"""
    integrate_thimbles(S, S_grad, domain, deformation_parameters, prefactor, params, mode)

Integrates the action function over the specified domain by deforming the integration contour along the steepest descent paths (thimbles) until convergence to the required accuracy 
or until the maximum number of flow steps is reached. This package is responsible for solving integrals of the form
\$\$
I = \\int^a_b f(z)e^{iS(z)}dz
\$\$
The parameters are listed as such:

| Parameter | Required | Type | Description |
| --------- | -------- | -------- | ----------- |
| `flow_steps` | Yes | `Int` | Maximum number of flow steps to perform. |
| `grid_spacing` | Yes | `Real` | Initial grad spacing parameters for the initial Lefschetz thimble contour. |
| `gradient_normalisation_factor` | Yes | `Real` | The normalisation threshold for the gradient during gradient descent. Used to prevent the gradient descent from diverging. |
| `flow_step_factor` | Yes | `Real` | The factor for determining how quickly the gradient descent moves. |
| `subdivision_threshold` | Yes | `Real` | The threshold value above which segments in the Lefschetz thimble contour are subdivided. |
| `height_threshold` | Yes | `Real` | The maximum magnitude of the imaginary component of the action to evolve the contour to. |
| `max_grid_element_count` | Yes | `Int` | The maximum number of simplices that can be used in the calculation of the Lefschetz thimble. |
| `verbose` | Yes | `Bool` | Whether or not to print the error message about when the simplices are maxed out. |
| `integral_accuracy` | Yes | `Real` | The required accuracy for the integral. |
| `integral_relative_error` | Yes | `Real` | The relative error threshold for the integral. |

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The gradient of the action function.
- `domain::Vector{RealDomain}`: The initial integration domain.
- `deformation_parameters::Vector{<:Number}`: The parameters used to deform the integration contour.
- `prefactor::Function`: A function that provides the prefactor for the integrand.
- `params::Dict`: Integration and flow parameters (e.g., `flow_steps`, `grid_spacing`, `gradient_normalisation_factor`, `flow_rate_scaling`, `integral_accuracy`, etc.).
- `mode::String`: The integration mode, i.e. "fixed" or "adaptive".

# Returns
- The result of the total integration over the flowed contours.
"""
function integrate_thimbles(S::Function, S_grad::Function, domain::Vector{RealDomain}, deformation_parameters::Vector{<:Number}, prefactor::Function, params::Dict, mode::String)
    flow_steps = params["flow_steps"]
    grid_spacing = params["grid_spacing"]
    gradient_normalisation_threshold = params["gradient_normalisation_threshold"]
    flow_step_factor = params["flow_step_factor"]
    subdivision_threshold = params["subdivision_threshold"]
    height_threshold = params["height_threshold"]
    max_grid_element_count = params["max_grid_element_count"]
    verbosity = params["verbose"]
    integral_accuracy = params["integral_accuracy"]
    integral_relative_error = params["integral_relative_error"]
    if length(domain) == 2
        # 2D case
        if mode == "fixed"
            init_points = Methods2D.Utils.make_init_points_rectangle(domain[1].min, domain[1].max, domain[2].min, domain[2].max)
            return Methods2D.Quadrilateral.DownwardsFlow.integrate_flowed_quads_fixed_Nflow(S, S_grad,
                init_points, prefactor=prefactor,
                Nflow=flow_steps, Δinit=grid_spacing,
                gradnthreshold=gradient_normalisation_threshold,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                h_threshold=height_threshold,
                maxNsimplices=max_grid_element_count,
                print_message=verbosity
            )
        else
            return Methods2D.Quadrilateral.DownwardsFlow.integrate_flowed_quads(S, S_grad,
                domain[1].min, domain[1].max,
                deformation_parameters[1],
                deformation_parameters[2],
                prefactor=prefactor,
                Nflow=flow_steps,
                Δinit=grid_spacing,
                gradnthreshold=gradient_normalisation_threshold,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                h_threshold=height_threshold,
                maxNsimplices=max_grid_element_count,
                integral_accuracy=integral_accuracy,
                integral_rel_error=integral_relative_error,
                print_message=verbosity
            )
        end
    elseif length(domain) == 1
        # 1D case
        return Methods1D.Integration.integrate_thimble(S, S_grad,
            domain[1].min, domain[1].max,
            prefactor=prefactor, Δinit=grid_scaping,
            flowstepfactor=flow_step_factor,
            gradnthreshold=gradient_normalisation_threshold,
            subdividethreshold=subdivision_threshold,
            h_threshold=height_threshold,
            Nmax=flow_steps,
            integral_accuracy=integral_accuracy,
            integral_rel_error=integral_relative_error,
            print_message=verbosity
        )
    end
end

# Integrate around all contributing thimbles, using the saddle point method, where individual saddle point contributions are summed over.
export integrate_thimbles
"""
    integrate_thimbles(S, S_grad, S_hessian, domain, params, prefactor; check)

Integrates around all contributing thimbles using the saddle point approximation method. The contributions from the individual contributing saddle points are summed up to give the total integral.
The parameters for algorithm are listed as below:

| Parameter | Required | Type | Description |
| --------- | -------- | ---- | ----------- |
| `output_dim` | Yes | `Int` | The dimension of the integral. |

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `domain::Vector{ComplexDomain}`: The domain over which to search for saddle points and integrate.
- `params::Dict`: Parameters for saddle point finding and integration (e.g., `output_dim`).
- `prefactor::Function`: A function that provides the prefactor for the integrand.
- `check::Function`: A function used to check if two found saddle points are identical (default `!isequal`).

# Returns
- The total integral approximated using the saddle point method.
"""
function integrate_thimbles(
    S::Function, S_grad::Function,
    S_hessian::Function,
    domain::Vector{ComplexDomain},
    params::Dict, prefactor::Function;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)
    if length(domain) == 1
        # 1D case
        return Methods1D.SaddlePoint.integrate_SPM(S, S_grad, S_hessian, domain[1].min, domain[1].max, prefactor=prefactor)
    elseif length(domain) == 2
        return _integrate_SPM(S, S_grad, S_hessian, domain, params, prefactor, check=check)
    end
end

function _integrate_SPM(
    S::Function, S_grad::Function,
    S_hessian::Function,
    domain::Vector{ComplexDomain},
    params::Dict, prefactor::Function;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)

    output_dim = params["output_dim"]

    saddles = find_numerical_saddles(S_grad, domain, params)
    total_integral = zeros(ComplexF64, output_dim)
    for saddle in saddles
        if check_contribution!(S, S_grad, S_hessian, saddle, domain[1], params)
            total_integral += Methods2D.SaddlePoint.saddles_gaussian_contribution(S, S_hessian, saddle, prefactor=prefactor)
        end
    end

    return total_integral
end

include("../wrappers/integration.jl")

end