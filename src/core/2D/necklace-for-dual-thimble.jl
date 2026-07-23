module DualThimble

using ..Types: FlowPoint, Simplex, Saddle
using LinearAlgebra
using StaticArrays

import Base.imag, Base.real
imag(p::FlowPoint) = FlowPoint(imag.(p.coords[1]), imag.(p.coords[2]), p.active)
real(p::FlowPoint) = FlowPoint(real.(p.coords[1]), real.(p.coords[2]), p.active)

imag(ls::Simplex{2,Int}) = Simplex{2,Int}(imag(points[ls.vertices[1]]), imag(points[ls.vertices[2]]), ls.active)
real(ls::Simplex{2,Int}) = Simplex{2,Int}(real(points[ls.vertices[1]]), real(points[ls.vertices[2]]), ls.active)

function norm(ls::Simplex{2,Int})
    norm([points[ls.vertices[2]].coords[1], points[ls.vertices[2]].coords[2]] .- [points[ls.vertices[1]].coords[1], points[ls.vertices[1]].coords[2]])
end

dist(p1::FlowPoint, p2::FlowPoint) = norm(@SVector[p2.coords[1] - p1.coords[1], p2.coords[2] - p1.coords[2]])
import Base.length
length(ls::Simplex{2,Int}) = dist(points[ls.vertices[2]], points[ls.vertices[1]])


function get_point(ls::Simplex{2,Int}, which::Symbol=:s)
    if which == :s
        return FlowPoint(points[ls.vertices[1]].coords[1], points[ls.vertices[1]].coords[2])
    elseif which == :e
        return FlowPoint(points[ls.vertices[2]].coords[1], points[ls.vertices[2]].coords[2])
    else
        return println("You've got a problem!")
    end
end

### this sorts linesegments such that they form a connected line
function sort_linesegs(linesegs::Vector{Simplex{2,Int}})
    linesegs_dict = Dict()
    for ls in linesegs
        linesegs_dict[ls.vertices[1]] = ls
    end

    # Initialize the sorted line segments vector with the first line segment
    sorted_linesegs = [linesegs[1]]

    # Iterate until all line segments are sorted
    while length(sorted_linesegs) != length(linesegs)
        # Find the next line segment based on the end Simplex{2, Int} of the last sorted segment
        matching_ls = linesegs_dict[last(sorted_linesegs).vertices[2]]
        push!(sorted_linesegs, matching_ls)
    end

    return sorted_linesegs
end

function adorn_necklace(necklace::Vector{Simplex{2,Int}}, points::Vector{<:FlowPoint})
    new_necklace = deepcopy(necklace)
    for i in eachindex(new_necklace)
        new_necklace[i] = Simplex{2,Int}(points[new_necklace[i].vertices[1]], points[new_necklace[i].vertices[2]])
    end

    return new_necklace
end

function make_quad(ls1::Simplex{2,Int}, ls2::Simplex{2,Int})
    p1 = points[ls1.vertices[1]]
    p2 = points[ls2.vertices[1]]
    p3 = points[ls1.vertices[2]]
    p4 = points[ls2.vertices[2]]
    return Simplex{4,FlowPoint}([p1, p2, p3, p4])
end

function make_quads(necklace::Vector{Simplex{2,Int}}, points::Vector{<:FlowPoint}, prev_necklace::Vector{Simplex{2,Int}})
    quads = Vector{Simplex{4,FlowPoint}}()
    new_necklace = adorn_necklace(sort_linesegs(deepcopy(necklace)), points)
    for simplex in eachindex(prev_necklace)
        quad = make_quad(prev_necklace[simplex], new_necklace[simplex])
        push!(quads, quad)
    end

    return quads
end

### simple Gauss area formula to find the area enclosed by the necklace (to double-check if it's not got folded into itself)
function enclosed_area(linesegs::Vector{Simplex{2,Int}}, f::Function=x -> real(x))
    # Initialize the area accumulator
    area = 0.0

    # Iterate over each line segment
    for i in 1:length(linesegs)
        # Get the coordinates of the endpoints of the line segment
        x1 = f(points[linesegs[i].vertices[1]].coords[1])
        y1 = f(points[linesegs[i].vertices[1]].coords[2])
        x2 = f(points[linesegs[i].vertices[2]].coords[1])
        y2 = f(points[linesegs[i].vertices[2]].coords[2])

        # Update the area accumulator
        area += x1 * y2 - x2 * y1
    end

    # Divide the result by 2 to get the absolute area
    area = abs(area) / 2.0

    return area
end


### necklacy things
function initialise!(necklace::Vector{Simplex{2,Int}}, points::Vector{FlowPoint},
    ti::ComplexF64, tr::ComplexF64,
    f::Function;
    Ninit::Int64=20,
    ϵ::Float64=0.01)

    hessian = FiniteDiff.finite_difference_hessian(
        tvec -> imag(f(complex(tvec[1:2]...), complex(tvec[3:4]...))),
        [reim(ti)..., reim(tr)...])
    # f_hessian(ti, tr) #my_hessian(b,Ip,q,ti,tr)
    # 
    # this could certainly be made more julian    
    eigenvectors = [[complex(vec[1:2]...), complex(vec[3:4]...)] for vec in eachcol(eigvecs(hessian))]

    # eigenvectors 3 and 4 are the ones with positive sign. So if I want the steepest ascent thimble, then I should use those.
    pointsini = ([[ti, tr] .+ ϵ * (cos(θ) * eigenvectors[3] + sin(θ) * eigenvectors[4]) for θ in range(0, stop=2π, length=Ninit + 1)])[1:end-1]
    # because 0 and 2π are the same and I don't want the point twice, me stupid!!!

    push!(points, [FlowPoint(p[1], p[2]) for p in pointsini]...)
    push!(necklace, [Simplex{2,Int}(i, i + 1, true) for i in 1:(length(points)-1)]...)
    push!(necklace, Simplex{2,Int}(length(points), 1, true)) # closing the necklace
end

### TODO this Δ could definitely get a more sophisticated default value
function subdivide!(simplex::Simplex{2,Int},
    necklace::Vector{Simplex{2,Int}}, points::Vector{FlowPoint};
    Δ::Float64=1.)

    p1 = points[simplex.vertices[1]]
    p2 = points[simplex.vertices[2]]

    active = p1.active && p2.active

    Δx(p1::FlowPoint, p2::FlowPoint) = norm(p2.coords[1] - p1.coords[1])
    Δy(p1::FlowPoint, p2::FlowPoint) = norm(p2.coords[2] - p1.coords[2])
    midx(p1::FlowPoint, p2::FlowPoint) = (p2.coords[1] + p1.coords[1]) ./ 2
    midy(p1::FlowPoint, p2::FlowPoint) = (p2.coords[2] + p1.coords[2]) ./ 2

    if active && (max(Δx(p1, p2), Δy(p1, p2)) > Δ)
        simplex.active = false # simplex gets turned inactive when being divided.       
        midpoint = FlowPoint(midx(p1, p2), midy(p1, p2))
        push!(points, midpoint)

        lineseg1mid = Simplex{2,Int}(simplex.vertices[1], length(points), true)
        linesegmid2 = Simplex{2,Int}(length(points), simplex.vertices[2], true)
        push!(necklace, lineseg1mid)
        push!(necklace, linesegmid2)
    end
end

### flowing up
function flow!(necklace::Vector{Simplex{2,Int}}, points::Vector{FlowPoint},
    f::Function,
    f_grad::Function;
    δ::Float64=0.1,
    threshold::Float64=0.5
)

    for i in 1:length(points)
        # TODO check both real and imaginary part?
        if points[i].active # for the active points
            # set them to be active (= still flowing) if they are above threshold
            # points[i].active = real(-im * S(b, Ip, points[i].coords[1], points[i].coords[2], q)) < 0 #(in Job's code that's h-function > thresh, I should clearly state which sign I'm using where etc.) 
            points[i].active = real(f(points[i].coords[1], points[i].coords[2])) < 0 #(in Job's code that's h-function > thresh, I should clearly state which sign I'm using where etc.) 

            if points[i].active
                step = δ .* gradN((ti, tr) -> conj.(complex.(f_grad(ti, tr))), points[i].coords[1], points[i].coords[2], threshold)
                points[i].coords[1] += step[1]
                points[i].coords[2] += step[2]
            end
        end
    end
end

function adorn_necklace!(necklace::Vector{Simplex{2,Int}}, points::Vector{FlowPoint})
    adorned = Vector{Simplex{2,FlowPoint}}()
    for i in 1:length(necklace)
        push!(adorned, Simplex{2,FlowPoint}([points[necklace[i].vertices[1]], points[necklace[i].vertices[2]]]))
    end
    return adorned
end

### get necklace
function get_necklace_solver(f::Function,
    f_grad::Function,
    f_hessian::Function,
    ti::ComplexF64, tr::ComplexF64;
    Ninit::Int64=20, Ncounter::Int64=600,
    eigvecfactorinit::Float64=0.04, # I should come up with sophisticated guesses here.
    flowstepfactor::Float64=0.4,
    subdividethreshold::Float64=1.8, kwargs...)

    necklace = Vector{Simplex{2,Int}}()
    points = Vector{FlowPoint}()

    initialise!(necklace, points, ti, tr, f, Ninit=Ninit, ϵ=eigvecfactorinit)

    necklaces = Vector{Vector{Simplex{2,Int}}}()
    push!(necklaces, adorn_necklace(sort_linesegs(deepcopy(necklace)), points))

    ### find a suitable threshold for the normalisation of the gradient
    gradient0 = [norm(conj.(f_grad(p.coords[1], p.coords[2]))) for p in points]
    threshold = round(minimum(gradient0), RoundDown, sigdigits=2)

    counter = 0
    quadrangles = Vector{Simplex{4,FlowPoint}}()

    while counter < Ncounter
        counter += 1

        tmp = deepcopy(necklace)
        flow!(necklace, points, f, f_grad, threshold=threshold, δ=flowstepfactor)

        if count([p.active for p in points]) == 0
            @debug "I broke because the flow stopped after $counter iterations"
            # println("I broke because the flow stopped after $counter iterations")
            break
        end

        prev_necklace = necklaces[end]
        new_quads = make_quads(deepcopy(necklace), points, prev_necklace)
        push!(quadrangles, new_quads...)
        push!(necklaces, adorn_necklace(sort_linesegs(deepcopy(necklace)), points))

        for i in 1:length(necklace)
            subdivide!(necklace[i], necklace, points, Δ=subdividethreshold)
        end
        keepat!(necklace, [ls.active for ls in necklace])
    end

    if counter == Ncounter && Ncounter > 1
        println("I broke because the counter reached its max, i.e. $Ncounter.")
    end

    necklace = sort_linesegs(necklace)
    adorned_necklace = adorn_necklace!(necklace, points)

    return adorned_necklace, quadrangles
end


function get_necklace(f::Function,
    f_grad::Function,
    f_hessian::Function,
    saddle_point::Saddle;
    logerrors::Bool=false,
    kwargs... # this passes on all th ekeyword arguments
)

    ti = saddle_point.saddle[1].coords[1]
    tr = saddle_point.saddle[2].coords[1]

    necklace, quadrangles = get_necklace_solver(f, f_grad, f_hessian, ti, tr; kwargs...)

    necklace_init, quadrangles_init = get_necklace_solver(f, f_grad, f_hessian, ti, tr; kwargs..., Ncounter=1)
    enclosed_area_init = enclosed_area(necklace_init, imag) + enclosed_area(necklace_init, real)

    if (enclosed_area(necklace, imag) + enclosed_area(necklace, real)) > enclosed_area_init
        return necklace, quadrangles
    else
        if (real(f(ti, tr))) > -0.2
            return necklace, quadrangles
        else
            @warn ("Warning (3)! The necklace is smaller than its initialisation, real(f) = $(real(f(ti, tr)))")
            # println("Warning (3)! The necklace is smaller than its initialisation for beam $b at q $q with ti $ti and tr $tr, where h was $(real(-im * S(b, Ip, ti, tr, q)))!")
            # logerrors ? log_error("necklace-errors.txt", "Warning (3) for beam $b at q $q with ti $ti and tr $tr.") : nothing
            return nothing, nothing
        end

    end


    # counter = 0
    # while ((enclosed_area(necklace,imag) + enclosed_area(necklace,real)) < 0.5) && 
    #     counter < 4 && length(necklace) > Ninit+1

    #     necklace = get_necklace_solver(b, Ip, q, ti, tr; Ninit = Ninit, Ncounter=Ncounter,
    #     eigvecfactorinit = eigvecfactorinit*2, # I should come up with sophisticated guesses here.
    #     flowstepfactor = flowstepfactor, 
    #     subdividethreshold = subdividethreshold )

    #     if counter==3 && logerrors
    #         log_error("necklace-errors.txt", "Warning (1) for beam $b at q $q with ti $ti and tr $tr.")
    #     end

    #     Ncounter *= 2 # random other guess to improve the necklace finding
    #     counter += 1
    # end
    # return necklace
end

end