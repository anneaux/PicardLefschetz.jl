module PathFlow

using LinearAlgebra, FastGaussQuadrature
using FiniteDiff

using ..Types: FlowPoint, Simplex, Saddle

function subdivide(points::Vector{<:FlowPoint},
    simplices::Vector{Simplex{2,Int}},
    Δ::Float64)

    for i in eachindex(simplices)
        sim = simplices[i]
        if sim.active
            l = sim.vertices[1]
            r = sim.vertices[2]
            L = points[l].coords[1]
            R = points[r].coords[1]
            if abs(R - L) > Δ
                push!(points, FlowPoint((L + R) / 2.))
                simplices[i].active = false
                push!(simplices, Simplex{2,Int}([l, length(points)]))
                push!(simplices, Simplex{2,Int}([length(points), r]))
            end
        end
    end
end


function subdivide_rep(points::Vector{<:FlowPoint},
    simplices::Vector{Simplex{2,Int}},
    δ::Float64)
    n_old = length(simplices)
    n_new = n_old + 1

    while n_old != n_new
        n_old = n_new
        subdivide(points, simplices, δ)
        n_new = length(simplices)
    end

    filter!(sim -> sim.active, simplices)
end

function initialise(tmin::Float64, tmax::Float64,
    Δ::Real, endpoints=[true, true])

    points = [FlowPoint(complex(tmin)), FlowPoint(complex(tmax))]
    points[1].active = endpoints[1]
    points[2].active = endpoints[2]
    simplices = [Simplex{2,Int}([1, 2])]

    subdivide_rep(points, simplices, Float64(Δ))
    filter!(sim -> sim.active, simplices)

    return (points, simplices)
end

function grad(drv::Function, t::ComplexF64)
    g = drv(t)
    return conj(complex(1im * g))
end

function gradN(drv::Function, t::ComplexF64, thresh::Float64=1.)
    g = grad(drv, t)
    if norm(g) > thresh # bit lower than the gradient at the saddle point
        return LinearAlgebra.normalize(g)
    else
        return g
    end
end

function flow_step(drv::Function, vertices::Complex, δ::Float64, direction::Float64, threshold::Float64=1.0)
    return vertices + direction * δ * gradN(drv, vertices, threshold)
end

function flow_line(S::Function, S_prime::Function, start_coord::Complex, δ::Float64, sign::Float64, should_stop::Function, flow_steps::Int)::Vector{FlowPoint}
    points = Vector{FlowPoint}()
    vertices = start_coord
    push!(points, FlowPoint(vertices))

    flow_count = 0
    while !should_stop(vertices) && flow_count < flow_steps
        vertices = flow_step(S_prime, vertices, δ, sign, 1.0)
        push!(points, FlowPoint(vertices))
        flow_count += 1
    end
    return points
end

function flow_down!(fun::Tuple,
    points::Vector{<:FlowPoint}, simplices::Vector{Simplex{2,Int}};
    δ::Float64=0.5, # flowstepfactor
    threshold::Float64=0.5, # for normalisation of thr gradient
    h_threshold::Float64=-20.
)

    S = fun[1]
    drv = fun[2]

    for i1 in 1:length(points)
        if points[i1].active # for the active points
            points[i1].coords[1] = flow_step(drv, points[i1].coords[1], δ, -1.0, threshold)
        end
    end

    for i2 in eachindex(simplices)
        if simplices[i2].active
            for v1 in simplices[i2].vertices
                if real(im * S(points[v1].coords[1])) < h_threshold
                    simplices[i2].active = false
                    points[v1].active = false
                end
            end
        end
    end
end

function get_hessian_eigenvectors!(directions::Vector{ComplexF64}, saddle_point::FlowPoint, S::Function, sign::Symbol)::Nothing
    # Step 1: Find the eigenvectors of the Hessian matrix evaluated at the saddle point.
    X = [real(saddle_point.coords[1]), imag(saddle_point.coords[1])]
    hessian = FiniteDiff.finite_difference_hessian(tvec -> imag(S(complex(tvec[1], tvec[2]))), X)
    eigen_decomposition = eigen(hessian)
    eigenvalues = eigen_decomposition.values
    eigenvectors = eigen_decomposition.vectors

    for i in eachindex(eigenvalues)
        if sign == :descent
            if eigenvalues[i] > 0
                v = complex(eigenvectors[:, i][1], eigenvectors[:, i][2])
                push!(directions, v)
                push!(directions, -v)
            end
        else
            if eigenvalues[i] < 0
                v = complex(eigenvectors[:, i][1], eigenvectors[:, i][2])
                push!(directions, v)
                push!(directions, -v)
            end
        end
    end
end

function find_intersection_point(thimble::Vector{FlowPoint})::FlowPoint
    prev_coord = thimble[end-1].coords[1]
    last_coord = thimble[end].coords[1]
    m = (imag(last_coord.coords[1]) - imag(prev_coord.coords[1])) / (real(last_coord.coords[1]) - real(prev_coord.coords[1]))
    c = imag(last_coord.coords[1]) - m * real(last_coord.coords[1])

    return FlowPoint(complex(-c / m, 0.0))
end

export flow_up
function flow_up(S::Function, S_prime::Function, saddle_point::FlowPoint, δ::Float64, h_threshold::Float64, flow_steps::Int)::Tuple{Vector{Vector{FlowPoint}},Bool}
    contributing = true
    thimbles = Vector{Vector{FlowPoint}}()
    max_height = abs(h_threshold)
    directions = Vector{ComplexF64}()
    get_hessian_eigenvectors!(directions, saddle_point, S, :ascent)

    on_real_line = imag(saddle_point.coords[1]) == 0

    if on_real_line
        # Both directions move away from the real line.
        if imag(directions[1]) > 0
            dir_up = directions[1]
            dir_down = directions[2]
        else
            dir_up = directions[2]
            dir_down = directions[1]
        end
        pass_up = saddle_point.coords[1] + δ * dir_up
        pass_down = saddle_point.coords[1] + δ * dir_down

        # Flow up to infinity (where -Im(S) exceeds max_height)
        should_stop_infinity(vertices) = -imag(S(vertices)) >= max_height
        points_up = flow_line(S, S_prime, pass_up, δ, 1.0, should_stop_infinity, flow_steps)
        points_down = flow_line(S, S_prime, pass_down, δ, 1.0, should_stop_infinity, flow_steps)

        push!(thimbles, points_up)
        push!(thimbles, points_down)
        contributing = true
    else
        # Determine forward (towards real line) and backward (away from real line) passes
        pass_1 = saddle_point.coords[1] + δ * directions[1]
        pass_2 = saddle_point.coords[1] + δ * directions[2]

        if abs(imag(pass_1)) < abs(imag(saddle_point.coords[1]))
            forward_pass = pass_1
            backward_pass = pass_2
        else
            forward_pass = pass_2
            backward_pass = pass_1
        end

        # Flow forward pass
        should_stop_forward(vertices) = (imag(vertices) * imag(saddle_point.coords[1]) <= 0) || (-imag(S(vertices)) >= max_height)
        points_forward = flow_line(S, S_prime, forward_pass, δ, 1.0, should_stop_forward, flow_steps)

        # If it crossed the real line, it contributes.
        last_coord = points_forward[end].coords[1]
        contributing = imag(last_coord) * imag(saddle_point.coords[1]) <= 0

        # Flow backward pass to infinity
        should_stop_infinity_backwards(vertices) = -imag(S(vertices)) >= max_height
        points_backward = flow_line(S, S_prime, backward_pass, δ, 1.0, should_stop_infinity_backwards, flow_steps)

        push!(thimbles, points_forward)
        push!(thimbles, points_backward)
    end

    if contributing && !on_real_line
        for i in length(thimbles[1]):1
            return thimbles, contributing, find_intersection_point(thimbles[1])
        end
    end

    return thimbles, contributing
end


export get_thimble
function get_thimble(S::Function, drv::Function, tmin::Float64, tmax::Float64;
    Nflow::Int64=60,
    Δinit::Real=10.,
    flowstepfactor::Real=2.,
    h_threshold::Real=-300.,
    gradnthreshold::Real=1.,
    subdividethreshold::Real=4.
)

    (points, simplices) = initialise(real(tmin), real(tmax), Δinit)

    for i_flow in 1:Nflow
        flow_down!((S, drv), points, simplices,
            threshold=gradnthreshold, δ=flowstepfactor, h_threshold=h_threshold)
        subdivide_rep(points, simplices, subdividethreshold)
    end

    filter!(sim -> sim.active, simplices)

    return points, simplices
end

export get_thimble
function get_thimble(S::Function, S_grad::Function, S_hessian::Function,
    saddle_point::Saddle; init_perturbation_radius::Float64,
    max_iterations::Int64, flow_step_factor::Float64,
    gradient_normalisation_threshold::Float64, subdivision_threshold::Float64,
    height_threshold::Float64
)

    saddle_mypt = saddle_point.saddle
    saddle_mypt.active = false

    directions = Vector{ComplexF64}()
    get_hessian_eigenvectors!(directions, saddle_mypt, S, :descent)

    dir1 = directions[1]
    dir2 = directions[2]

    pt_left = FlowPoint(saddle_mypt.coords[1] + init_perturbation_radius * dir1)
    pt_right = FlowPoint(saddle_mypt.coords[1] + init_perturbation_radius * dir2)

    points = [saddle_mypt, pt_left, pt_right]

    simplices = [Simplex{2,Int}([2, 1]), Simplex{2,Int}([1, 3])]

    for i_flow in 1:max_iterations
        flow_down!(
            (S, S_grad),
            points,
            simplices,
            δ=flow_step_factor,
            threshold=gradient_normalisation_threshold,
            h_threshold=height_threshold
        )

        subdivide_rep(points, simplices, subdivision_threshold)
    end

    filter!(sim -> sim.active, simplices)

    return points, simplices
end


function dissect_thimbles(points, simplices)
    active_linesegs = filter(sim -> sim.active, simplices)

    thimbles = Vector()
    visited = falses(length(active_linesegs))


    function dfs!(active_linesegs, visited, i_start)
        stack = [i_start] # "open ends" to explore (= the two point indices of the start line segment)
        trace = Int[]
        while !isempty(stack)
            v = pop!(stack)

            if !visited[v]
                visited[v] = true
                push!(trace, v)
                ### find the Simplex{2, Int} of the next linesegment
                next = findall(ls -> ls.vertices[1] == active_linesegs[v].vertices[2], active_linesegs)
                append!(stack, next)
                prev = findall(ls -> ls.vertices[2] == active_linesegs[v].vertices[1], active_linesegs)
                append!(stack, prev)
            end
        end
        return trace
    end

    for i in 1:length(active_linesegs)
        if !visited[i]
            trace = dfs!(active_linesegs, visited, i)
            push!(thimbles, active_linesegs[trace])
        end
    end

    return thimbles
end

end