module Saddle

using ..Methods1D
using ..Methods2D
using ..Types

export find_analytic_saddles
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
function find_numerical_saddles(
    derivative::Function,
    domain::Vector{ComplexDomain},
    params::Dict;
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)

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
function check_contribution!(
    S::Function,
    S_grad::Function,
    S_hessian::Function,
    saddle_point::Types.Saddle,
    domain::ComplexDomain,
    params::Dict;
    log_errors::Bool=false
)
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

end