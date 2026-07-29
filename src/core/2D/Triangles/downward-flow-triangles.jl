module DownwardsFlow

using ...Types: FlowPoint, Simplex
using StaticArrays

#### OTHER UTILS
Base.getindex(z::Iterators.Zip, i) = (it -> getindex(it, i)).(z.is)
toR4(z1::ComplexF64, z2::ComplexF64) = [real(z1), imag(z1), real(z2), imag(z2)]
toR4(z1::Number, z2::Number) = [real(z1), imag(z1), real(z2), imag(z2)]


function project_onto_triangle(base::AbstractVector, points::AbstractVector) # what are these types?
    @assert length(base) == 3 "Need exactly 3 vertices"

    # Convert to R^4
    vs = [toR4(z.x, z.y) for z in base]

    # Basepoint
    v1, v2, v3 = vs
    e1 = v2 - v1
    e2 = v3 - v1

    # Gram-Schmidt to find orthonormal basis
    u1 = e1 / norm(e1)
    e2_proj = e2 - (u1 ⋅ e2) * u1
    u2 = e2_proj / norm(e2_proj)

    # Projection function
    proj(v) = [(u1 ⋅ (v - v1)), (u2 ⋅ (v - v1))]

    # Project all points
    ps = [proj(toR4(p[1], p[2])) for p in points]
    gradient0 = [norm(conj.(f_grad(p[1], p[2]))) for p in points]

    return ps
end

using MiniQhull
flags_original = "qhull d Qt Qbb Qc Qz"

prod(points::AbstractVector) = (points[2][1] - points[1][1]) * (points[3][2] - points[1][2]) - (points[2][2] - points[1][2]) * (points[3][1] - points[1][1])


function triangle_normal(p1, p2, p3)
    return cross(
        [p2[1], p2[2], 0.] - [p1[1], p1[2], 0.],
        [p3[1], p3[2], 0.] - [p1[1], p1[2], 0.]
    )
end




function subdivide_triangle_new!(points, t_vertices::Simplex{3,Int};
    Δ::Real=0.5, delaunay_flags=flags_original)
    # og_angle = my_angle(prod(points[t_vertices.vertices]))
    n_ref = triangle_normal(points[t_vertices.vertices]...)

    vertices_here = Vector{Int}()
    push!(vertices_here, t_vertices.vertices...)
    v1, v2, v3 = t_vertices.vertices
    t_edges = [(v1, v2), (v2, v3), (v3, v1)] ### indices of the specific one I'm looking at
    for k in t_edges
        p1, p2 = points[[k...]]
        d = dist(p1, p2)
        if d > Δ ### subdivide

            ### how many points to insert in between?
            ### wanna make sure that the new distance is not larger than Δ
            n = floor(Int64, d / Δ)
            xvals = range(p1[1], stop=p2[1], length=n + 2)
            yvals = range(p1[2], stop=p2[2], length=n + 2)
            new_points = [FlowPoint(c...) for c in zip(xvals[2:end-1], yvals[2:end-1])] # I don't want to include the actual points themselves

            for np in new_points
                already_created = findall(p -> isequal(round.(xy(p), digits=8), round.(xy(np), digits=8)), points)
                if isempty(already_created)
                    push!(points, np)
                    push!(vertices_here, length(points))
                    #         println("point is new. new Simplex{2, Int}: $(length(points)),  length points: $(length(points))")
                else
                    #             println("point is already created. length points: $(length(points))")
                    np_vertex = already_created[1]
                    push!(vertices_here, np_vertex)
                    ### add a pointer to the existing point
                end
            end

        end
    end

    if length(vertices_here) == 3
        return points, Vector{Simplex{3,Int}}[]
    else
        ### project only the points that are relevant for this new triangle
        projected_points = project_onto_triangle(points[t_vertices.vertices], points[vertices_here])
        connections_here = delaunay(projected_points, flags_original) ### they are in the Simplex{2, Int} system of this triangle!

        ### check that they are not degenerate!         
        connections_here_vectors = [collect(col) for col in eachcol(connections_here)]
        filter!(ind -> triangle_area_new(projected_points[ind]) > 1e-5,
            connections_here_vectors)

        # new_triangles = [Simplex{3, Int}(vertices_here[inds]) for inds in connections_here_vectors]
        new_triangles = Simplex{3,Int}[]
        ### the following is just to ensure that the newly created triangles have the same orientation
        for inds in connections_here_vectors
            verts = vertices_here[inds]
            n = triangle_normal(points[verts]...)

            if real(dot(n, n_ref)) > 0
                push!(new_triangles, Simplex{3,Int}(verts))
            else
                push!(new_triangles, Simplex{3,Int}(reverse(verts)))
            end
        end
        return points, new_triangles
    end
end


function dissect_thimbles(triangles)
    active_triangles = filter(s -> s.active, triangles)
    thimbles = Vector()
    visited = falses(length(active_triangles))

    function dfs!(active_triangles, visited, i_start)
        stack = [i_start] # "open ends" to explore (= the four corner point indices of the quadrilateral)
        trace = Int[] # trace is more like a carpet now I guess
        while !isempty(stack)
            v = pop!(stack)

            if !visited[v]
                # #                 @show v
                visited[v] = true
                push!(trace, v)

                ### find the neighbouring quads to the vertices
                for vertex in 1:3
                    # find all quads that corner to you
                    nexts = findall(sim -> in(active_triangles[v].vertices[vertex], sim.vertices), active_triangles)
                    #                     @show nexts
                    append!(stack, filter(!isequal(v), nexts)) # obviously ignore the quad that you're in atm
                end
                unique!(stack)
            end
        end
        return trace
    end

    for i in 1:length(active_triangles)
        #     i = 1
        if !visited[i]
            trace = dfs!(active_triangles, visited, i)
            push!(thimbles, active_triangles[trace])
        end
    end

    return thimbles
end


function triangle_area_new(points::AbstractVector)
    @assert length(points) == 3
    abs((points[2][1] - points[1][1]) * (points[3][2] - points[1][2]) - (points[2][2] - points[1][2]) * (points[3][1] - points[1][1])) / 2
end
# triangle_area(points::AbstractVector{<:FlowPoint}) = triangle_area([xy(p) for p in points])


#### DOWNWARDS FLOW

function orient_triangle(verts::AbstractArray, points::Vector{SVector{2,Float64}}, ref_angle::Float64)
    if dot(ref_angle, angle(prod(points[verts]))) > 0
        verts
    else
        reverse(verts)
    end
end


function initialise_grid_triangles(points::Vector{<:FlowPoint}, Δ::Float64)

    pts = [SVector(real(p[1]), real(p[2])) for p in points]
    connections = delaunay(pts) #, "qhull d Qbb Qc QJ Pp")
    #     "qhull d Qbb Qc QJ Pp"
    #         display(connections)
    triangles = [Simplex{3,Int}(orient_triangle(inds, pts, +100.)) for inds in eachcol(connections)]
    points = [float2complex(p) for p in points]

    ### filter out degenerate points!
    n_old = length(triangles)
    n_new = n_old + 1

    i_while = 0
    while (n_old != n_new)
        n_old = n_new
        i_while += 1
        i = 1

        for i_t in eachindex(triangles)
            triangle = triangles[i_t]
            if triangle.active
                #                 points_this_triangle = points[triangle.vertices]
                points, new_connections = subdivide_triangle_new!(points, triangle, Δ=Δ)
                if !isempty(new_connections)
                    triangle.active = false
                    append!(triangles, new_connections)
                end

            end
            if i > 30
                break
            end
        end

        if i_while > 10
            break
        end
        filter!(sim -> sim.active, triangles)
        n_new = length(triangles)

    end
    return points, triangles
end

#### not sure what this is actually
# function initialise_triangles_on_real_plane(xmin, xmax, ymin, ymax, Δx, Δy)
#     xrange = range(real(xmin), stop = real(xmax), length = Int(ceil(real(xmax-xmin)/Δx)))
#     yrange = range(real(ymin), stop = real(ymax), length = Int(ceil(real(ymax-ymin)/Δy)))

#     points_ini = Vector{<:FlowPoint}()
#     for xr in collect(xrange)
#         for yr in collect(yrange)
#             push!(points_ini, FlowPoint(xr, yr,true))
#         end
#     end

#     pts = [SVector(p.x, p.y) for p in points_ini]
#     connections = delaunay(pts) #, "qhull d Qbb Qc QJ Pp")
#     #     "qhull d Qbb Qc QJ Pp"
# 	#         display(connections)

# 	### filter out degenerate points!

#     #### THIS REQUIRES FURTHER DEVELOPMENT

#     return points_ini, connections
# end

### this flows the whole surface
function flow_down!(triangles::Vector{Simplex{3,Int}}, points::Vector{<:FlowPoint},
    f::Function,
    f_grad::Function;
    threshold::Real=0.5, # for normalisation of the gradient
    δ::Real=0.5, # flowstepfactor
    h_threshold::Real=-20.
)

    for i1 in 1:length(points)
        if points[i1].active # for the active points
            step = -δ .* gradN((ti, tr) -> conj.(complex.(f_grad(ti, tr))), points[i1][1] + 0im, points[i1][2] + 0im, threshold)
            points[i1].coords[1] += step[1]
            points[i1].coords[2] += step[2]

            ### turning them inactive when they are below the threshold. Doing this here prevents lonely relict points from flowing if their triangle has turned inactive already.
            if real(f(xy(points[i1])...)) < h_threshold
                points[i1].active = false
            end
        end
    end

    for i2 in eachindex(triangles)
        if triangles[i2].active
            for v in triangles[i2].vertices
                if !points[v].active
                    # if real(f(points[v].x, points[v].y)) < h_threshold 
                    triangles[i2].active = false # am I sure that I want to turn the whole simplex inactive?
                    # points[v].active = false
                end
            end
        end
    end
end


function subdivide_triangles!(points::Vector{<:FlowPoint}, triangles::Vector{Simplex{3,Int}}, subdividethreshold::Real)
    n_old = length(triangles)
    n_new = n_old + 1

    i_while = 0
    while (n_old != n_new)
        n_old = n_new
        i_while += 1
        # i = 1

        ### in subdivide sub routine ideally
        for i_t in eachindex(triangles)
            triangle = triangles[i_t]
            if triangle.active
                if all([p.active for p in points[triangle.vertices]])
                    points, new_connections = subdivide_triangle_new!(points, triangle, Δ=subdividethreshold)
                    # n_actives = sum([t.active for t in triangles])
                    if !isempty(new_connections)
                        triangle.active = false
                        append!(triangles, new_connections)
                    end
                elseif !any([p.active for p in points[triangle.vertices]])
                    triangle.active = false
                    # println("here i have turned a triangle inactive")
                end
            end
        end

        #  if i_while > 10 break end
        filter!(sim -> sim.active, triangles)
        n_new = length(triangles)
    end
end

### requires integrate triangles.jl


### get_flowed_triangles()
# to flow from the original real-valued domain, and return Simplex{3, FlowPoint}
export get_flowed_triangles
function get_flowed_triangles(
    f::Function,
    f_grad::Function,
    init_points::Vector{<:FlowPoint};
    Nflow::Int64=50,
    Δinit::Float64=10.,
    gradnthreshold::Float64=0.05, # grad normalisation threshold
    flowstepfactor::Float64=2., # flowstepfactor
    subdividethreshold::Float64=8., # subdivide threshold, wants to be 4 * δ
    h_threshold::Float64=-150.,
    maxNsimplices::Int64=5000,
    tolNsimplices::Float64=0.05,
    flow_bounds::Vector{Bool}=[true, true, true, true]
)

    netsimplices = Vector{Int64}()
    (points, simplices) = initialise_grid_triangles(init_points, Δinit)

    overboard = false
    push!(netsimplices, length(simplices))

    for i_flow in 1:Nflow
        nsimplices = length(simplices)
        # println(netsimplices)
        flow_down!(simplices, points, f, f_grad,
            threshold=gradnthreshold, δ=flowstepfactor, h_threshold=h_threshold)
        subdivide_triangles!(points, simplices, subdividethreshold)

        ### TODO convergence criteria
        #         net = (length(simplices)-nsimplices)
        #         # println(net)
        #         if net !== 0
        #             push!(netsimplices, net)
        #             i_flow > 1 ? overboard = true : false
        #         # elseif (net) == 0 && overboard
        #         #     println("I naturally converged at flow step $i_flow !") ; break
        #         end

        #         if has_converged(netsimplices[2:end], tol = tolNsimplices*netsimplices[1]) 
        #             println("I converged w.r.t. number of simplices at i_flow = $i_flow."); 
        #             break 
        #         end

        #         if isempty(findall(sim->sim.active, simplices))
        #             println("I broke because all things finished flowing after $i_flow steps."); break
        #         end
        #         if length(simplices) > maxNsimplices
        #             println("I broke after $i_flow steps because I have more than $maxNsimplices simplices now."); break
        #         end        

        if i_flow == Nflow
            println("I stopped because I reached the maximum flow steps, i.e. $Nflow.")
        end
    end

    triangles = [Simplex{3,FlowPoint}(points[sim.vertices]) for sim in simplices]

    return triangles, points, simplices
end

end