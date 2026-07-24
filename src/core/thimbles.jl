module Thimble

using ..Methods1D
using ..Methods2D
using ..Types
using ..Saddle: find_numerical_saddles

# Gets the thimble for a given saddle point.
export get_thimble!
function get_thimble!(
    S::Function, S_grad::Function, S_hessian::Function,
    saddle_point::Types.Saddle, params::Dict;
    mesh_type::String
)::Nothing
    thimble = if length(saddle_point.saddle) == 2
        initial_point_count = params["initial_point_count"]
        max_iterations = params["max_iterations"]
        init_perturbation_radius = params["init_perturbation_radius"]
        flow_step_factor = params["flow_step_factor"]
        subdivision_threshold = params["subdivision_threshold"]
        if mesh_type == "quad"
            accuracy = params["accuracy"]
            necklace, quads = Methods2D.Quadrilateral.Thimble.get_SD_thimble_quads(
                S, S_grad, S_hessian,
                saddle_point,
                Ninit=initial_point_count,
                Ncounter=max_iterations,
                accuracy=accuracy,
                eigvecfactorinit=init_perturbation_radius,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold
            )
            quads
        elseif mesh_type == "triangles"
            gradient_normalisation_threshold = params["gradient_normalisation_threshold"]
            height_threshold = params["height_threshold"]
            Methods2D.Triangle.Thimble.get_SD_thimble_triangles(
                S, S_grad, S_hessian,
                saddle_point,
                Ninit=initial_point_count,
                Nflow=max_iterations,
                eigvecfactorinit=init_perturbation_radius,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                gradn_threshold=gradient_normalisation_threshold,
                h_threshold=height_threshold
            )
        else
            throw(error("Mesh type not supported."))
        end
    elseif length(saddle_point.saddle) == 1
        init_perturbation_radius = params["init_perturbation_radius"]
        max_iterations = params["max_iterations"]
        flow_step_factor = params["flow_step_factor"]
        subdivision_threshold = params["subdivision_threshold"]
        gradient_normalisation_threshold = params["gradient_normalisation_threshold"]
        height_threshold = params["height_threshold"]
        Methods1D.PathFlow.get_thimble(
            S, S_grad, S_hessian, saddle_point,
            init_perturbation_radius=init_perturbation_radius,
            max_iterations=max_iterations,
            flow_step_factor=flow_step_factor,
            subdivision_threshold=subdivision_threshold,
            gradient_normalisation_threshold=gradient_normalisation_threshold,
            height_threshold=height_threshold
        )
    end

    saddle_point.thimble = thimble
    return nothing
end

# Gets the thimbles for all saddle points.
export get_thimbles
function get_thimbles(
    S::Function, S_grad::Function,
    params::Dict, domain::Vector{RealDomain},
    contributing::Bool; mesh_type::String
)
    # TODO: check for contributing saddle points when returning output.
    if length(domain) == 1
        iterations = params["iterations"]
        init_subdivision = params["init_subdivision"]
        flow_step_factor = params["flow_step_factor"]
        height_threshold = params["height_threshold"]
        gradient_normalisation_threshold = params["gradient_normalisation_threshold"]
        subdivision_threshold = params["subdivision_threshold"]
        return Methods1D.PathFlow.get_thimble(S, S_grad, domain[1].min, domain[1].max,
            Nflow=iterations, Δinit=init_subdivision,
            flowstepfactor=flow_step_factor, h_threshold=height_threshold,
            gradnthreshold=gradient_normalisation_threshold, subdividethreshold=subdivision_threshold
        )
    elseif length(domain) == 2
        flow_steps = params["flow_steps"]
        grid_resolution = params["grid_resolution"]
        gradient_normalisation_threshold = params["gradient_normalisation_threshold"]
        subdivision_threshold = params["subdivision_threshold"]
        height_threshold = params["height_threshold"]
        max_simplices = params["max_simplices"]
        simplex_tolerance = params["simplex_tolerance"]
        flow_step_factor = params["flow_step_factor"]

        init_points = Methods2D.Utils.make_init_points_rectangle(domain[1].min, domain[1].max, domain[2].min, domain[2].max)

        if mesh_type == "quad"
            return Methods2D.Quadrilateral.DownwardsFlow.get_flowed_quads(
                S, S_grad, init_points,
                Nflow=flow_steps,
                Δinit=grid_resolution,
                gradnthreshold=gradient_normalisation_threshold,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                h_threshold=height_threshold,
                maxNsimplices=max_simplices,
                tolNsimplices=simplex_tolerance
            )
        elseif mesh_type == "triangles"
            return Methods2D.Triangles.DownwardsFlow.get_flowed_triangles(
                S, S_grad, init_points,
                Nflow=flow_steps,
                Δinit=grid_resolution,
                gradnthreshold=gradient_normalisation_threshold,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                h_threshold=height_threshold,
                maxNsimplices=max_simplices,
                tolNsimplices=simplex_tolerance
            )
        else
            throw(error("Mesh type not supported."))
        end
    else
        throw(error("Input vector unsupported length."))
    end
end

# Gets the thimble boundary for a given saddle point.
export get_thimble_boundary!
function get_thimble_boundary!(
    S::Function, S_grad::Function, S_hessian::Function,
    saddle_point::Types.Saddle, params::Dict;
    mesh_type::String
)::Nothing
    init_perturbation_radius = params["init_perturbation_radius"]
    max_iterations = params["max_iterations"]
    flow_step_factor = params["flow_step_factor"]
    subdivision_threshold = params["subdivision_threshold"]
    gradient_normalisation_threshold = params["gradient_normalisation_threshold"]
    height_threshold = params["height_threshold"]

    boundary = if length(saddle_point.saddle) == 1
        points, simplices = Methods1D.PathFlow.get_thimble(S, S_grad, S_hessian,
            saddle_point,
            init_perturbation_radius=init_perturbation_radius,
            max_iterations=max_iterations,
            flow_step_factor=flow_step_factor,
            subdivision_threshold=subdivision_threshold,
            gradient_normalisation_threshold=gradient_normalisation_threshold,
            height_threshold=height_threshold
        )

        counts = Dict{Int,Int}()
        for simplex in simplices
            for index in simplex.vertices
                counts[index] = get(counts, index, 0) + 1
            end
        end
        boundary_indices = [index for (index, count) in counts if count == 1]
        [points[index].coords for index in boundary_indices]
    elseif length(saddle_point.saddle) == 2
        init_point_count = params["init_point_count"]
        accuracy = params["accuracy"]
        init_perturbation_radius = params["init_perturbation_radius"]

        if mesh_type == "quad"
            necklace, quads = Methods2D.Quadrilateral.Thimble.get_SD_thimble_quads(
                S, S_grad, S_hessian,
                saddle_point,
                Ninit=init_point_count,
                Ncounter=max_iterations,
                accuracy=accuracy,
                eigvecfactorinit=init_perturbation_radius,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold
            )
            necklace
        elseif mesh_type == "triangle"
            necklace, trianglesC, points_all, triangles = Methods2D.Triangle.Thimble.get_SD_thimble_triangles(
                S, S_grad, S_hessian,
                saddle_point,
                Ninit=init_point_count,
                Nflow=max_iterations,
                eigvecfactorinit=init_perturbation_radius,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                gradn_threshold=gradient_normalisation_threshold,
                h_threshold=height_threshold
            )
            necklace
        end
    end

    saddle_point.thimble_boundary = boundary
    return nothing
end

# Gets the thimble boundary for all saddle points.
export get_thimble_boundaries
function get_thimble_boundaries(
    S::Function, S_grad::Function, S_hessian::Function,
    domain::Vector{ComplexDomain}, params::Dict,
    contributing::Bool; mesh_type::String,
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Saddle}
    saddles = find_numerical_saddles(S_grad, domain, params)
    for saddle in saddles
        if contributing
            check_contribution!(S, S_grad, S_hessian, saddle, domain[1], params)
            if saddle.contributing
                get_thimble_boundary!(S, S_grad, S_hessian, saddle, params, mesh_type=mesh_type)
            end
        else
            get_thimble_boundary!(S, S_grad, S_hessian, saddle, params, mesh_type=mesh_type)
        end
    end

    return saddles
end

include("../wrappers/thimbles.jl")

end