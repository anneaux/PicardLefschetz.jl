module PathFlow

using LinearAlgebra, FastGaussQuadrature
using FiniteDiff

mutable struct MyPoint
    coord::Complex
    active::Bool
    MyPoint(coord) = new(coord, true)
end

mutable struct Index
    coord::Vector{Int}
    active::Bool
    Index(coord) = new(coord, true)
end

function subdivide(points::Vector{MyPoint},
    simplices::Vector{Index},
    Δ::Float64)

    for i in eachindex(simplices)
        sim = simplices[i]
        if sim.active
            l = sim.coord[1]
            r = sim.coord[2]
            L = points[l].coord
            R = points[r].coord
            if abs(R - L) > Δ
                push!(points, MyPoint((L + R) / 2.))
                simplices[i].active = false
                append!(simplices, [Index([l, length(points)]), Index([length(points), r])])
            end
        end
    end
end


function subdivide_rep(points::Vector{MyPoint},
    simplices::Vector{Index},
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
    Δ::Float64, endpoints=[true, true])

    points = [MyPoint(tmin), MyPoint(tmax)]
    points[1].active = endpoints[1]
    points[2].active = endpoints[2]
    simplices = [Index([1, 2])]

    subdivide_rep(points, simplices, Δ)
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

function flow_step(drv::Function, coord::Complex, δ::Float64, direction::Float64, threshold::Float64=1.0)
    return coord + direction * δ * gradN(drv, coord, threshold)
end

function flow_line(S::Function, S_prime::Function, start_coord::Complex, δ::Float64, sign::Float64, should_stop::Function, flow_steps::Int)::Vector{MyPoint}
    points = Vector{MyPoint}()
    coord = start_coord
    push!(points, MyPoint(coord))

    flow_count = 0
    while !should_stop(coord) && flow_count < flow_steps
        coord = flow_step(S_prime, coord, δ, sign, 1.0)
        push!(points, MyPoint(coord))
        flow_count += 1
    end
    return points
end

function flow_down!(fun::Tuple,
    points::Vector{MyPoint}, simplices::Vector{Index};
    δ::Float64=0.5, # flowstepfactor
    threshold::Float64=0.5, # for normalisation of thr gradient
    h_threshold::Float64=-20.
)

    S = fun[1]
    drv = fun[2]

    for i1 in 1:length(points)
        if points[i1].active # for the active points
            points[i1].coord = flow_step(drv, points[i1].coord, δ, -1.0, threshold)
        end
    end

    for i2 in eachindex(simplices)
        if simplices[i2].active
            for v1 in simplices[i2].coord
                if real(im * S(points[v1].coord)) < h_threshold
                    simplices[i2].active = false
                    points[v1].active = false
                end
            end
        end
    end
end

function get_hessian_eigenvectors!(directions::Vector{Complex64}, saddle_point::MyPoint, S::Function, sign::Symbol)::Nothing
    # Step 1: Find the eigenvectors of the Hessian matrix evaluated at the saddle point.
    X = [real(saddle_point.coord), imag(saddle_point.coord)]
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

function find_intersection_point(thimble::Vector{MyPoint})::MyPoint
    prev_coord = thimble[end-1].coord
    last_coord = thimble[end].coord
    m = (imag(last_coord.coord) - imag(prev_coord.coord)) / (real(last_coord.coord) - real(prev_coord.coord))
    c = imag(last_coord.coord) - m * real(last_coord.coord)

    return MyPoint(complex(-c / m, 0.0))
end

function flow_up(S::Function, S_prime::Function, saddle_point::MyPoint, δ::Float64, h_threshold::Float64, flow_steps::Int)::Tuple{Vector{Vector{MyPoint}},Bool}
    contributing = true
    thimbles = Vector{Vector{MyPoint}}()
    max_height = abs(h_threshold)
    directions = Vector{ComplexF64}()
    get_hessian_eigenvectors!(directions, saddle_point, S, :ascent)

    on_real_line = imag(saddle_point.coord) == 0

    if on_real_line
        # Both directions move away from the real line.
        if imag(directions[1]) > 0
            dir_up = directions[1]
            dir_down = directions[2]
        else
            dir_up = directions[2]
            dir_down = directions[1]
        end
        pass_up = saddle_point.coord + δ * dir_up
        pass_down = saddle_point.coord + δ * dir_down

        # Flow up to infinity (where -Im(S) exceeds max_height)
        should_stop_infinity(coord) = -imag(S(coord)) >= max_height
        points_up = flow_line(S, S_prime, pass_up, δ, 1.0, should_stop_infinity, flow_steps)
        points_down = flow_line(S, S_prime, pass_down, δ, 1.0, should_stop_infinity, flow_steps)

        push!(thimbles, points_up)
        push!(thimbles, points_down)
        contributing = true
    else
        # Determine forward (towards real line) and backward (away from real line) passes
        pass_1 = saddle_point.coord + δ * directions[1]
        pass_2 = saddle_point.coord + δ * directions[2]

        if abs(imag(pass_1)) < abs(imag(saddle_point.coord))
            forward_pass = pass_1
            backward_pass = pass_2
        else
            forward_pass = pass_2
            backward_pass = pass_1
        end

        # Flow forward pass
        should_stop_forward(coord) = (imag(coord) * imag(saddle_point.coord) <= 0) || (-imag(S(coord)) >= max_height)
        points_forward = flow_line(S, S_prime, forward_pass, δ, 1.0, should_stop_forward, flow_steps)

        # If it crossed the real line, it contributes.
        last_coord = points_forward[end].coord
        contributing = imag(last_coord) * imag(saddle_point.coord) <= 0

        # Flow backward pass to infinity
        should_stop_infinity(coord) = -imag(S(coord)) >= max_height
        points_backward = flow_line(S, S_prime, backward_pass, δ, 1.0, should_stop_infinity, flow_steps)

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



function get_thimble(S::Function, drv::Function, tmin::Float64, tmax::Float64;
    Nflow::Int64=60,
    Δinit::Float64=10.,
    flowstepfactor::Float64=2.,
    h_threshold::Float64=-300.,
    gradnthreshold::Float64=1.,
    subdividethreshold::Float64=4.
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
                ### find the index of the next linesegment
                next = findall(ls -> ls.coord[1] == active_linesegs[v].coord[2], active_linesegs)
                append!(stack, next)
                prev = findall(ls -> ls.coord[2] == active_linesegs[v].coord[1], active_linesegs)
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