### everything to decide whether or not a given saddle point contributes.
### this could be implemented in various methods again. Also maybe it should give a warning if there're multiple saddle points nearby and if a Gaussian approximation is a bad idea?
using ..Types: Simplex, FlowPoint, Saddle, convert_to_mesh
using ..DualThimble: get_necklace, get_point
import LinearAlgebra: norm


### utils for deciding whether a line crosses a given point
function distance_point_to_line(p::AbstractVector, s::AbstractVector, t::AbstractVector)
    midpoint = (s .+ t) ./ 2
    return norm(p .- midpoint)
end

function distance_point_to_line(p::FlowPoint, l::Simplex{2,FlowPoint})
    return distance_point_to_line([p[1], p[2]], [l.vertices[1][1], l.vertices[1][2]], [l.vertices[2][1], l.vertices[2][2]])
end

norm(s::Simplex{2,FlowPoint}) = norm(s.vertices[1].coords .- s.vertices[2].coords)


# function find_crossing(line::Vector{Simplex{2, Int}}, point::FlowPoint, tolerance::Float64=0.8)
#     mindist, Simplex{2, Int} = findmin([distance_point_to_line(point, seg) for seg in line])

#     if mindist < tolerance
#         return Simplex{2, Int}
#     else 
#         return nothing
#     end
# end

function average_distance(line::Vector{Simplex{2,FlowPoint}}, pidx::Int64, threshold::Float64=0.5) # flowstepfactor
    line_region = [line[pidx]]

    for r in 1:min(10, length(line) - pidx - 1)
        push!(line_region, line[pidx+r])
        if norm(line[pidx+r]) > threshold # 2*flowstepfactor
            break
        end
    end
    for r in -1:-1:-min(10, pidx - 1)
        push!(line_region, line[pidx+r])
        if norm(line[pidx+r]) > threshold
            break
        end
    end

    av_dist = sum([norm(ls) for ls in line_region]) / length(line_region)#nregion
    #     @show av_dist
    return av_dist
end

function find_crossing(line::Vector{Simplex{2,FlowPoint}}, point::FlowPoint, tolerance::Float64=1.; threshold::Float64=0.5,
    loginfo=[])

    distances = [distance_point_to_line(point, seg) for seg in line]
    mindist, idx = findmin(distances)

    if mindist < tolerance && mindist < average_distance(line, idx, threshold)
        return idx
    else
        return nothing
    end
end


function find_crossing(curve::Contour.Curve2{Tuple{T,T}}, point::FlowPoint, tolerance::Float64=0.8; threshold::Float64=0.5) where T<:Real
    line = [Simplex{2,FlowPoint}(FlowPoint[FlowPoint(complex(curve.vertices[i][1]), complex(curve.vertices[i][2])), FlowPoint(complex(curve.vertices[i+1][1]), complex(curve.vertices[i+1][2]))]) for i in 1:(length(curve.vertices)-1)]
    return find_crossing(line, point, tolerance, threshold=threshold)
end

function find_crossing(nocurve::Missing, point::FlowPoint, tolerance::Float64=0.8; threshold::Float64=0.5)
    return nothing
end

# ### calculating the contour line through a given saddle
# function real_projected_contourlines(
#     f::Function,
#     ti::ComplexF64, tr::ComplexF64,
#     ti_range::Real=30, tr_range::Real=50
#     ; Ntimes = 101)    

#     # TC = TCycle(b)
#     tir_values = range(real(ti)- ti_range, stop = real(ti) + ti_range, length = Ntimes)
#     # tii_values = range(-1., stop = imag(ti) + 0.25TC, length = Ntimes)
#     trr_values = range(max(real(tr)- tr_range, real(ti) + ti_range+0.1) , stop = real(tr) + tr_range, length = Ntimes)
#     # tri_values = range(-1., stop = imag(ti) + 0.25TC, length = Ntimes)  # this is wrong, because tr can have negative imaginary part! Luckily I don't need that here anyway ;-)

#     ### level line for the saddle point
#     # S_values = [-1im*S(b, Ip, complex(tir), complex(trr), q) for tir in tir_values, trr in trr_values]
#     # S_saddle = -1im*S(b, Ip, ti, tr, q)
#     S_values = [f(complex(tir), complex(trr)) for tir in tir_values, trr in trr_values]
#     S_saddle = f(ti, tr)
#     contour_saddle = Contour.contour(tir_values, trr_values, imag.(S_values), imag(S_saddle) )

#     return contour_saddle.lines
# end


### checking if conditions are fulfilled
function check_contribution(necklace::Vector{Simplex{2,FlowPoint}},
    f::Function,
    saddle_point::Saddle, check::Function
    ; kwargs...)

    flowstepfactor = try
        kwargs[:flowstepfactor]
    catch e
        0.8
    end

    intersection_point = find_intersection_for_contribution(necklace, saddle_point; flowstepfactor=flowstepfactor)

    H_at_hp = imag(f([necklace[idx].vertices[1].coords[1], necklace[idx].vertices[1].coords[2]]))
    H_at_sp = imag(f([ti, tr]))
    if abs(H_at_hp - H_at_sp) < 1. && (check([complex(intersection_point.vertices[1].coords[1]), complex(intersection_point.vertices[1].coords[2])]))
        return true
    end
end

function find_intersection_for_contribution(necklace::Vector{Simplex{2,FlowPoint}}, saddle_point::Saddle; flowstepfactor::Real)
    ti = saddle_point[1]
    tr = saddle_point[2]

    ### check if necklace hits real plane
    p = FlowPoint(0.0im, 0.0im)
    idx = find_crossing(imag.(necklace), p, threshold=2 * flowstepfactor) # can add loginfo here

    return necklace[idx]
end

function check_contribution(necklace::Nothing,
    f::Function,
    f_grad::Function,
    f_hessian::Function,
    saddle_point::Saddle,
    check::Function
    ; kwargs...)
    return false
end

function check_contribution(necklace::Nothing,
    f::Function,
    saddle_point::Saddle,
    check::Function
    ; kwargs...)
    return false
end

export check_contribution
function check_contribution(
    f::Function,
    f_grad::Function,
    f_hessian::Function,
    saddle_point::Saddle,
    check::Function,
    ; logerrors::Bool=false, kwargs...)
    # Ncounter = 600, logerrors::Bool=false)

    ti = saddle_point[1]
    tr = saddle_point[2]

    if real(f([ti, tr])) < 0
        necklace, _, points = get_necklace(f, f_grad, f_hessian, saddle_point; logerrors=logerrors, kwargs...)
        mesh = necklace === nothing ? nothing : convert_to_mesh((points, necklace))
        check_contribution(mesh, f, saddle_point, check)
    else
        @debug "it doesn't contribute! (0)"
        return false
    end
    # what happens if doesn't converge?   
end
