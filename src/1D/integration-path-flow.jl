

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

function subdivide_keep(points::Vector{MyPoint}, simplices::Vector{Index}, Δ::Float64)
    i = 1
    while i <= length(simplices)
        sim = simplices[i]
        if sim.active # only subdivide active ones
            l = sim.coord[1]
            r = sim.coord[2]
            L = points[l].coord
            R = points[r].coord
            if abs(R - L) > Δ
                push!(points, MyPoint((L + R) / 2.))
                new_idx = length(points)
                # Replace the parent simplex at index i with the first child
                simplices[i] = Index([l, new_idx])
                # Push the second child
                push!(simplices, Index([new_idx, r]))
                # Do not increment i, so we check the new first child again!
                continue
            end
        end
        i += 1
    end
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
end;

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
            step = -δ .* gradN(drv, points[i1].coord, threshold)
            points[i1].coord += step
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

function get_thimble(S::Function, drv::Function, tmin::Real, tmax::Real;
    preset::Symbol=:accurate,
    params::Union{Nothing,Dict}=nothing,
    omega::Float64=1.0,
    xi::Union{ComplexF64,Nothing}=nothing,
    keep_connected::Bool=false,
    promote_bridges::Bool=false,
    kwargs...
)
    resolved_params = if params !== nothing
        resolve_heuristics(params; kwargs...)
    else
        get_pl_heuristics_1d(S, drv, xi, omega, preset; tmin=tmin, tmax=tmax, kwargs...)
    end
    Nflow = resolved_params.Nflow
    Δinit = resolved_params.Δinit
    flowstepfactor = resolved_params.flowstepfactor
    h_threshold = resolved_params.h_threshold
    gradnthreshold = resolved_params.gradnthreshold
    subdividethreshold = resolved_params.subdividethreshold

    (points, simplices) = initialise(Float64(tmin), Float64(tmax), Δinit)

    for i_flow in 1:Nflow
        flow_down!((S, drv), points, simplices,
            threshold=gradnthreshold, δ=flowstepfactor, h_threshold=h_threshold)
        if keep_connected
            n_old = length(simplices)
            subdivide_keep(points, simplices, subdividethreshold)
            while length(simplices) != n_old
                n_old = length(simplices)
                subdivide_keep(points, simplices, subdividethreshold)
            end
        else
            subdivide_rep(points, simplices, subdividethreshold)
        end
    end

    if keep_connected
        simplify_inactive_segments!(points, simplices; promote_bridges=promote_bridges)
    else
        filter!(sim -> sim.active, simplices)
    end

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

function dissect_inactive_segments(points, simplices)
    inactive_linesegs = filter(sim -> !sim.active, simplices)

    inactive_groups = Vector()
    visited = falses(length(inactive_linesegs))

    function dfs!(inactive_linesegs, visited, i_start)
        stack = [i_start]
        trace = Int[]
        while !isempty(stack)
            v = pop!(stack)

            if !visited[v]
                visited[v] = true
                push!(trace, v)

                # Check for adjacent segments sharing either vertex in any orientation
                v1, v2 = inactive_linesegs[v].coord[1], inactive_linesegs[v].coord[2]
                next_adj = findall(ls -> ls.coord[1] == v2 || ls.coord[2] == v2 || ls.coord[1] == v1 || ls.coord[2] == v1, inactive_linesegs)
                append!(stack, next_adj)
            end
        end
        return trace
    end

    for i in 1:length(inactive_linesegs)
        if !visited[i]
            trace = dfs!(inactive_linesegs, visited, i)
            push!(inactive_groups, inactive_linesegs[trace])
        end
    end

    return inactive_groups
end

function has_adjacent_inactive_segments(points, simplices)
    inactive_components = dissect_inactive_segments(points, simplices)
    return !isempty(inactive_components)
end

function simplify_inactive_segments!(points, simplices; promote_bridges::Bool=false)
    # 1. Use dissect_thimbles to identify active thimble components
    active_components = dissect_thimbles(points, simplices)

    # 2. Build a map of vertex index -> active component ID
    point_to_comp = Dict{Int,Int}()
    for (comp_idx, comp_segs) in enumerate(active_components)
        for seg in comp_segs
            for v in seg.coord
                point_to_comp[v] = comp_idx
            end
        end
    end

    # 3. Find connected components of inactive segments
    inactive_components = dissect_inactive_segments(points, simplices)

    # 4. Remove all inactive segments from the simplices vector completely
    filter!(s -> s.active, simplices)

    # 5. Only add/keep the single simplified bridge for components that connect two different active thimbles
    for comp in inactive_components
        # Count the frequency of each vertex in the component
        vertex_counts = Dict{Int,Int}()
        for sim in comp
            for v in sim.coord
                vertex_counts[v] = get(vertex_counts, v, 0) + 1
            end
        end

        # Endpoints of the chain will appear exactly once
        endpoints = [v for (v, count) in vertex_counts if count == 1]

        if length(endpoints) == 2
            ep1, ep2 = endpoints[1], endpoints[2]

            # Look up which active thimbles these endpoints connect to
            c1 = get(point_to_comp, ep1, 0)
            c2 = get(point_to_comp, ep2, 0)

            # Only insert the single bridge if they connect two DIFFERENT active components
            if c1 != 0 && c2 != 0 && c1 != c2
                new_sim = Index([ep1, ep2])
                if promote_bridges
                    new_sim.active = true
                    points[ep1].active = true
                    points[ep2].active = true
                else
                    new_sim.active = false
                end
                push!(simplices, new_sim)
            end
        end
    end
    return simplices
end


function mapping(p, p1, p2)
    return (p1 * (1. - p[1]) + p2 * (1. + p[1])) / 2.
end
function jacobian(p, p1, p2)
    return (p2 - p1) / 2.
end

function integrate_line(integrand::Function, line, n, lattice, weights)
    sum = 0
    for i = 1:n
        sum = sum + weights[i] * integrand(lattice[i], line[1], line[2])
    end
    return sum
end

function integrate_line(integrand::Function, line, n::Int64=7)
    lattice, weights = gausslegendre(n)
    return integrate_line(integrand, line, n, lattice, weights)
end

function integrate_thimble(S::Function,
    points::Vector, simplices::Vector;
    prefactor::Function=t -> 1.)
    points_r = map(pp -> pp.coord, points)
    simplices_r = map(sim -> sim.coord, simplices)

    n = 7
    lattice, weights = gausslegendre(n)

    function integrand(pp, p1, p2)
        return prefactor(mapping(pp, p1, p2)) * jacobian(pp, p1, p2) * exp(im * S(mapping(pp, p1, p2)))
    end

    sum = 0.
    for i in eachindex(simplices_r)
        sum = sum + integrate_line(integrand, points_r[simplices_r[i]], n, lattice, weights)
    end
    return sum

end

### fast direct integration of a flowed domain
function integrate_thimble(S::Function, drv::Function, tmin::Number, tmax::Number;
    prefactor::Function=t -> 1.,
    ### flow kwargs
    Δinit::Float64=10.,
    flowstepfactor::Float64=2.,
    h_threshold::Float64=-300.,
    gradnthreshold::Float64=1.,
    subdividethreshold::Float64=4.,
    ### convergence kwargs
    Nmax::Int64=50,
    integral_accuracy::Float64=1e-7,
    integral_rel_error::Float64=1e-3,
    print_message::Bool=true
)

    (points, simplices) = initialise(real(tmin), real(tmax), Δinit)
    prev_integral = complex(1.)
    int = complex(0.)

    for i_flow in 1:Nmax
        flow_down!((S, drv), points, simplices,
            threshold=gradnthreshold, δ=flowstepfactor, h_threshold=h_threshold)
        subdivide_rep(points, simplices, subdividethreshold)

        int = integrate_thimble(S, points, simplices, prefactor=prefactor)

        abs_diff = norm(int .- prev_integral)
        if 0 < abs_diff < integral_accuracy # if I don't want to stop here I can just set the goal incredibly low
            if print_message
                println("I broke after $i_flow iterations because the accuracy goal of $integral_accuracy was met, using $(length(simplices)) simplices.")
            end
            break
        end

        rel_diff = !isnan((int .- prev_integral) ./ prev_integral) ? norm((int .- prev_integral) ./ prev_integral) : -1.
        if 0 < rel_diff < integral_rel_error
            if print_message
                println("I broke after $i_flow iterations because the accuracy goal of $integral_rel_error relative error of the integral was met, using $(length(simplices)) simplices.")
            end
            break
        end

        prev_integral = int

        if i_flow == Nmax && print_message
            println("I stopped because I reached the maximum flow steps, i.e. $Nmax, with $(length(simplices)) simplices.")
        end

        #     filter!(sim->sim.active, simplices)
    end

    return int, points, simplices
end




