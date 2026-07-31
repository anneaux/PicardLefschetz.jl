module Thimble

using ..Methods1D
using ..Methods2D
using ..Types
using ..Saddle: find_saddles

# Gets the thimble for a given saddle point.
export get_thimble!
"""
    get_thimble!(S, S_grad, S_hessian, saddle_point, params; mesh_type)

Calculates the Lefschetz thimble for a given saddle point, using gradient descent. 
Gradient descent is done by solving the ODE along a time parameter \$ \\tau \$ to find the thimble. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = -\\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Always Required | Type | Description                                                                                                                                   | Heuristic |
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `flow_step_factor`    | Yes      | `Real` | The step size of the gradient descent solver.                                                                                                  | Yes | 
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 | Yes | 
| `height_threshold`    | Yes      | `Real` | The maximal magnitude of the imaginary component during gradient descent.        | Yes | 
| `gradient_normalisation_threshold` | Yes | `Real` | The threshold for normalising the gradient in the flow. | Yes |
| `init_perturbation_radius` | Yes  | `Real` | The initial radius for the Lefschetz thimble contour. This is the contour evolved to give the thimble. | Yes | 
| `subdivision_threshold` | Yes     | `Real` | The threshold for subdivding the segments between the points of the Lefschetz thimble. | Yes | 
| `init_point_count` | No       | `Int` | The initial number of points for the Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) | Yes | 
| `accuracy` | No | `Int` | The precision used during the thimble finding process. (This parameter is required when `mesh_type` is "quad"). | No | 

# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `saddle_point::Saddle`: The struct containing the saddle point and the related functions.
- `params::Dict`: The parameters for the thimble calculation. All parameters listed above are to be included in this dictionary.
- `mesh_type::String`: The type of mesh to generate in 2D (either "quad" or "triangles").

# Returns
- `Nothing`
"""
function get_thimble!(
    S::Function, S_grad::Function, S_hessian::Function,
    saddle_point::Types.Saddle, params::Dict;
    mesh_type::String
)::Nothing
    init_perturbation_radius = Float64(params["init_perturbation_radius"])
    max_iterations = params["max_iterations"]
    flow_step_factor = Float64(params["flow_step_factor"])
    subdivision_threshold = Float64(params["subdivision_threshold"])

    thimble = if length(saddle_point.saddle) == 2
        initial_point_count = params["init_point_count"]
        if mesh_type == "quad"
            accuracy = params["accuracy"]
            necklace, quads, points = Methods2D.Quadrilateral.Thimble.get_SD_thimble_quads(
                S, S_grad, S_hessian,
                saddle_point,
                Ninit=initial_point_count,
                Ncounter=max_iterations,
                accuracy=accuracy,
                eigvecfactorinit=init_perturbation_radius,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold
            )
            (points, quads)
        elseif mesh_type == "triangles"
            gradient_normalisation_threshold = Float64(params["gradient_normalisation_threshold"])
            height_threshold = Float64(params["height_threshold"])
            necklace, trianglesC, points_all, triangles = Methods2D.Triangle.Thimble.get_SD_thimble_triangles(
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
            (points_all, triangles)
        else
            throw(error("Mesh type not supported."))
        end
    elseif length(saddle_point.saddle) == 1
        gradient_normalisation_threshold = Float64(params["gradient_normalisation_threshold"])
        height_threshold = Float64(params["height_threshold"])
        points, simplices = Methods1D.PathFlow.get_thimble(
            S, S_grad, S_hessian, saddle_point,
            init_perturbation_radius=init_perturbation_radius,
            max_iterations=max_iterations,
            flow_step_factor=flow_step_factor,
            subdivision_threshold=subdivision_threshold,
            gradient_normalisation_threshold=gradient_normalisation_threshold,
            height_threshold=height_threshold
        )

        (points, simplices)
    end

    saddle_point.thimble = thimble
    return nothing
end

# Gets the thimbles for all saddle points.
export get_FLIC
"""
    get_FLIC(S, S_grad, params, domain; mesh_type)

Calculates the Lefschetz thimbles for a given domain, using gradient descent on an initial grid of points over the domain. 
Gradient descent is done by solving the ODE along a time parameter \$ \\tau \$ to find the thimble. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = -\\overline{\\nabla S(z)}
\$\$
The total contour returned is the flowed integration domain forming the thimbles, i.e. a connected version of the Lefschetz thimbles which are obtained by using gradient descent from the real space.

The parameters for this function are listed in the table:

| Parameter             | Always Required | Type | Description                                                                                                                                   | Heuristic | 
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ----- | 
| `flow_step_factor`    | Yes      | `Real` | The step size of the gradient descent solver.                                                                                                  | Yes |
| `height_threshold`    | Yes      | `Real` | The maximal magnitude of the imaginary component during gradient descent.        | Yes | 
| `gradient_normalisation_threshold` | Yes | `Real` | The threshold for normalising the gradient in the flow. | Yes | 
| `subdivision_threshold` | Yes     | `Real` | The threshold for subdivding the segments between the points of the Lefschetz thimble. | Yes |
| `max_iterations` | Yes | `Int` | The maximum number of iterations to perform using the solver. | Yes | 
| `init_point_count` | No | `Int` | The initial number of points in the Lefschetz thimble contour, only required in 2D. | Yes |
| `max_simplices` | No | `Int` | The maximum number of simplices (quads or triangles) to allow during the flow in 2D. | No | 
| `simplex_tolerance` | No | `Int` | The tolerance for the number of simplices during the flow in 2D. | No |
| `promote_bridges` | No | `Bool` | Whether to promote the bridges between thimbles to active thimbles. (This parameter is only required in 1D). | No | 
| `keep_connected` | No | `Bool` | Whether to keep the thimbles connected. (This parameter is only required in 1D). | No |

# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `params::Dict`: The parameters for the thimble calculation. All parameters listed above are to be included in this dictionary.
- `domain::Vector{RealDomain}`: The domain over which to calculate the thimbles. In the 1D case, the domain should be only one `RealDomain`, and in 2D, it should be `RealDomain`s.
- `mesh_type::String`: The type of mesh to generate in 2D (either "quad" or "triangles").

# Returns
- `Tuple{Vector{<:FlowPoint}, Vector{<:Simplex}}`: The result of the flow (a set of segments in 1D, or a list of quadrilaterals/triangles in 2D).
"""
function get_FLIC(
    S::Function, S_grad::Function,
    params::Dict, domain::Vector{RealDomain}; mesh_type::String
)
    max_iterations = params["max_iterations"]
    gradient_normalisation_threshold = Float64(params["gradient_normalisation_threshold"])
    subdivision_threshold = Float64(params["subdivision_threshold"])
    height_threshold = Float64(params["height_threshold"])
    flow_step_factor = Float64(params["flow_step_factor"])
    if length(domain) == 1
        keep_connected = params["keep_connected"]
        promote_bridges = params["promote_bridges"]
        return Methods1D.PathFlow.get_thimble(S, S_grad, domain[1].min, domain[1].max,
            Nflow=max_iterations, Δinit=subdivision_threshold,
            flowstepfactor=flow_step_factor, h_threshold=height_threshold,
            gradnthreshold=gradient_normalisation_threshold, subdividethreshold=subdivision_threshold,
            promote_bridges=promote_bridges, keep_connected=keep_connected
        )
    elseif length(domain) == 2
        init_point_count = Float64(params["init_point_count"])
        max_simplices = params["max_simplices"]
        simplex_tolerance = params["simplex_tolerance"]

        init_points = Methods2D.Utils.make_init_points_rectangle(domain[1].min, domain[1].max, domain[2].min, domain[2].max)

        if mesh_type == "quad"
            quads, points, simplices = Methods2D.Quadrilateral.DownwardsFlow.get_flowed_quads(
                S, S_grad, init_points,
                Nflow=max_iterations,
                Δinit=init_point_count,
                gradnthreshold=gradient_normalisation_threshold,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold,
                h_threshold=height_threshold,
                maxNsimplices=max_simplices,
                tolNsimplices=simplex_tolerance
            )

            return points, simplices
        elseif mesh_type == "triangles"
            triangles, points, simplices = Methods2D.Triangles.DownwardsFlow.get_flowed_triangles(
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

            return points, simplices
        else
            throw(error("Mesh type not supported."))
        end
    else
        throw(error("Input vector unsupported length."))
    end
end

export get_thimbles

"""
    get_thimbles(S, S_grad, S_hessian, params, domain, contributing; mesh_type)

Calculates the Lefschetz thimbles for all saddle points, using gradient descent to be able to calculate the thimble. 
Gradient descent is done by solving the ODE along a time parameter \$ \\tau \$ to find the thimble. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = -\\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Always Required | Type | Description                                                                                                                                   | Heuristic | 
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------ | 
| `flow_step_factor`    | Yes      | `Real` | The step size of the gradient descent solver.                                                                                                  | Yes | 
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 | Yes |
| `height_threshold`    | Yes      | `Real` | The maximal magnitude of the imaginary component during gradient descent. This is used when checking if the saddle point contributes.        | Yes | 
| `gradient_normalisation_threshold` | Yes | `Real` | The threshold for normalising the gradient in the flow. | Yes | 
| `init_perturbation_radius` | Yes  | `Real` | The initial radius for the Lefschetz thimble contour. This is the contour evolved to give the thimble. | Yes | 
| `subdivision_threshold` | Yes     | `Real` | The threshold for subdivding the segments between the points of the Lefschetz thimble. | Yes | 
| `init_point_count` | No       | `Int` | The initial number of points for the Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) | Yes |
| `accuracy` | No | `Int` | The precision used during the thimble finding process. (This parameter is required when `mesh_type` is "quad"). | No |


# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `params::Dict`: The parameters for the thimble calculation. All parameters listed above are to be included in this dictionary.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the thimbles. In the 1D case, the domain should be only one `ComplexDomain`, and in 2D, it should be `ComplexDomain`s.
- `contributing::Bool`: Whether to include only contributing points. 
- `mesh_type::String`: The type of mesh to generate in 2D (either "quad" or "triangles").

# Returns
- `Vector{Saddle}`: A vector of the saddle points and their thimbles.
"""
function get_thimbles(S::Function, S_grad::Function, S_hessian::Function, params::Dict, domain::Vector{ComplexDomain}, contributing::Bool; mesh_type::String)::Vector{Saddle}
    saddles = find_saddles(S_grad, domain, params)
    for saddle in saddles
        if contributing
            check_contribution!(S, S_grad, S_hessian, saddle, domain[1], params)
            if saddle.contributing
                get_thimble!(S, S_grad, S_hessian, saddle, params, mesh_type=mesh_type)
            end
        else
            get_thimble!(S, S_grad, S_hessian, saddle, params, mesh_type=mesh_type)
        end
    end

    return saddles
end

# Gets the thimble boundary for a given saddle point.
export get_thimble_boundary!
"""
    get_thimble_boundary!(S, S_grad, S_hessian, saddle_point, params; mesh_type)

Calculates the boundary of the Lefschetz thimble for a given saddle point, using gradient descent. 
Gradient descent is done by solving the ODE along a time parameter \$ \\tau \$ to find the thimble boundary. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = -\\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | AlwaysRequired | Type | Description                                                                                                                                   | Heuristic | 
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------ | 
| `flow_step_factor`    | Yes      | `Real` | The step size of the gradient descent solver.                                                                                                  | Yes | 
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 | Yes | 
| `height_threshold`    | Yes      | `Real` | The maximal magnitude of the imaginary component during gradient descent.        | Yes | 
| `gradient_normalisation_threshold` | Yes | `Real` | The threshold for normalising the gradient in the flow. | Yes | 
| `init_perturbation_radius` | Yes  | `Real` | The initial radius for the Lefschetz thimble contour. This is the contour evolved to give the thimble. | Yes | 
| `subdivision_threshold` | Yes     | `Real` | The threshold for subdivding the segments between the points of the Lefschetz thimble. | Yes | 
| `init_point_count` | No       | `Int` | The initial number of points for the Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) | Yes | 
| `accuracy` | No | `Int` | The precision used during the thimble finding process. (This parameter is required when `mesh_type` is "quad"). | No | 

# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `saddle_point::Saddle`: The struct containing the saddle point and the related functions.
- `params::Dict`: The parameters for the thimble boundary calculation. All parameters listed above are to be included in this dictionary.
- `mesh_type::String`: The type of mesh to generate in 2D (either "quad" or "triangle").

# Returns
- `Nothing`
"""
function get_thimble_boundary!(
    S::Function, S_grad::Function, S_hessian::Function,
    saddle_point::Types.Saddle, params::Dict;
    mesh_type::String
)::Nothing
    init_perturbation_radius = Float64(params["init_perturbation_radius"])
    max_iterations = params["max_iterations"]
    flow_step_factor = Float64(params["flow_step_factor"])
    subdivision_threshold = Float64(params["subdivision_threshold"])
    gradient_normalisation_threshold = Float64(params["gradient_normalisation_threshold"])
    height_threshold = Float64(params["height_threshold"])

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
        T = typeof(points[1].coords)
        T[points[index].coords for index in boundary_indices]
    elseif length(saddle_point.saddle) == 2
        init_point_count = params["init_point_count"]
        accuracy = params["accuracy"]

        if mesh_type == "quad"
            necklace, quads, points = Methods2D.Quadrilateral.Thimble.get_SD_thimble_quads(
                S, S_grad, S_hessian,
                saddle_point,
                Ninit=init_point_count,
                Ncounter=max_iterations,
                accuracy=accuracy,
                eigvecfactorinit=init_perturbation_radius,
                flowstepfactor=flow_step_factor,
                subdividethreshold=subdivision_threshold
            )
            (points, necklace)
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
            (points_all, necklace)
        end
    end

    saddle_point.thimble_boundary = boundary
    return nothing
end

# Gets the thimble boundary for all saddle points.
export get_thimble_boundaries
"""
    get_thimble_boundaries(S, S_grad, S_hessian, domain, params, contributing; mesh_type, check)

Calculates the boundaries of the Lefschetz thimbles for all saddles, using gradient descent. 
Gradient descent is done by solving the ODE along a time parameter \$ \\tau \$ to find the thimble boundary. The ODE is as such
\$\$
\\frac{dS}{d\\tau} = -\\overline{\\nabla S(z)}
\$\$

The parameters for this function are listed in the table:

| Parameter             | Always Required | Type | Description                                                                                                                                   | Heuristic | 
| --------------------- | -------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ------ | 
| `flow_step_factor`    | Yes      | `Real` | The step size of the gradient descent solver.                                                                                                  | Yes | 
| `max_iterations`      | Yes      | `Int` | The maximum number of iterations to perform using the solver.                                                                                 | Yes | 
| `height_threshold`    | Yes      | `Real` | The maximal magnitude of the imaginary component during gradient descent.        | Yes | 
| `gradient_normalisation_threshold` | Yes | `Real` | The threshold for normalising the gradient in the flow. | Yes | 
| `init_perturbation_radius` | Yes  | `Real` | The initial radius for the Lefschetz thimble contour. This is the contour evolved to give the thimble. | Yes | 
| `subdivision_threshold` | Yes     | `Real` | The threshold for subdivding the segments between the points of the Lefschetz thimble. | Yes | 
| `init_point_count` | No       | `Int` | The initial number of points for the Lefschetz thimble contour. (This parameter is only required when the dimension of the saddle is 2.) | Yes | 
| `accuracy` | No | `Int` | The precision used during the thimble finding process. (This parameter is required when `mesh_type` is "quad"). | No | 

# Arguments
- `S::Function`: The action function. 
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `domain::Vector{ComplexDomain}`: The domain over which to calculate the thimble boundaries.
- `params::Dict`: The parameters for the thimble calculation. All parameters listed above are to be included in this dictionary.
- `contributing::Bool`: Whether to include only contributing points.
- `mesh_type::String`: The type of mesh to generate in 2D (either "quad" or "triangle").
- `check::Function`: A function for checking equality of critical points.

# Returns
- `Vector{Saddle}`: A vector of the saddle points and their thimble boundaries.
"""
function get_thimble_boundaries(
    S::Function, S_grad::Function, S_hessian::Function,
    domain::Vector{ComplexDomain}, params::Dict,
    contributing::Bool; mesh_type::String,
    check::Function=(t_1, t_2) -> !isequal(t_1, t_2)
)::Vector{Saddle}
    saddles = find_saddles(S_grad, domain, params; check=check)
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