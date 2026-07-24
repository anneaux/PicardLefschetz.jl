module Integration

using ..Methods1D
using ..Methods2D
using ..Types
using ..Saddle

# Integrate using a given boundary around one thimble, i.e. boundary can be precomputed in this case.
export integrate_thimble
function integrate_thimble(S::Function, boundary::Vector, prefactor::Function, params::Dict)
    if boundary isa Vector{Any}
        # 1D case
        return Methods1D.Integration.integrate_thimble(S, boundary[1], boundary[2])
    elseif boundary isa Vector{QuadC}
        # 2D case
        integral = 0
        for quad in boundary
            integral += Methods2D.Quadrilateral.Integration.integrate_quadrilateral(S, quad, params["GL_order"], prefactor=prefactor)
        end

        return integral
    elseif boundary isa Vector{TriangleC}
        integral = 0
        for triangle in boundary
            integral += Methods2D.Triangle.Integration.integrate_triangle(S, triangle, prefactor=prefactor, order=params["simplex_order"], dim=params["output_dim"])
        end

        return integral
    end
end

# Integrate without a given boundary around one thimble, essentially steepest descent until integral converges to the required precision, or the number of flow steps is exceeded.
export integrate_thimble!
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
function integrate_thimbles(S::Function, S_grad::Function, domain::Vector{RealDomain}, deformation_parameters::Vector{<:Number}, prefactor::Function, params::Dict, mode::String)
    if length(domain) == 2
        # 2D fixed case
        if mode == "fixed"
            init_points = Methods2D.Utils.make_init_points_rectangle(domain[1].min, domain[1].max, domain[2].min, domain[2].max)
            return Methods2D.Quadrilateral.DownwardsFlow.integrate_flowed_quads_fixed_Nflow(S, S_grad,
                init_points, prefactor=prefactor,
                Nflow=params["flow_steps"], Δinit=params["grid_spacing"],
                gradnthreshold=params["gradient_normalisation_factor"],
                flowstepfactor=params["flow_rate_scaling"],
                subdividethreshold=params["subdivision_threshold"],
                h_threshold=params["height_threshold"],
                maxNsimplices=params["max_grid_element_count"],
                print_message=params["verbose"]
            )
        else
            return Methods2D.Quadrilateral.DownwardsFlow.integrate_flowed_quads(S, S_grad,
                domain[1].min, domain[1].max,
                deformation_parameters[1],
                deformation_parameters[2],
                prefactor=prefactor,
                Nflow=params["flow_steps"],
                Δinit=params["grid_spacing"],
                gradnthreshold=params["gradient_normalisation_factor"],
                flowstepfactor=params["flow_rate_scaling"],
                subdividethreshold=params["subdivision_threshold"],
                h_threshold=params["height_threshold"],
                maxNsimplices=params["max_grid_element_count"],
                integral_accuracy=params["integral_accuracy"],
                integral_rel_error=params["integral_relative_error"],
                print_message=params["verbose"]
            )
        end
    elseif length(domain) == 1
        # 1D case
        return Methods1D.Integration.integrate_thimble(S, S_grad,
            domain[1].min, domain[1].max,
            prefactor=prefactor, Δinit=params["grid_spacing"],
            flowstepfactor=params["flow_rate_scaling"],
            gradnthreshold=params["gradient_normalisation_factor"],
            subdividethreshold=params["subdivision_threshold"],
            h_threshold=params["height_threshold"],
            Nmax=params["max_flow_steps"],
            integral_accuracy=params["integral_accuracy"],
            integral_rel_error=params["integral_relative_error"],
            print_message=params["verbose"]
        )
    end
end

# Integrate around all contributing thimbles, using the saddle point method, where individual saddle point contributions are summed over.
export integrate_thimbles
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