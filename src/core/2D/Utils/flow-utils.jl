### normalised gradient

using ..Types: FlowPoint, Simplex

function gradN(
    f_grad::Function,
    ti::ComplexF64, tr::ComplexF64,
    thresh::Float64=1.)

    g = f_grad(ti, tr)
    if norm(g) > thresh # bit lower than the gradient at the saddle point
        return LinearAlgebra.normalize(g)
    else
        return g
    end
end;

xy(p::FlowPoint) = (p.coords[1], p.coords[2])

dist(p1::FlowPoint, p2::FlowPoint) = norm(p1.coords .- p2.coords)

function float2complex(p::FlowPoint)
    res = FlowPoint(complex(p.coords[1]), complex(p.coords[2]))
    res.active = p.active
    return res
end
toR4(p::FlowPoint) = toR4(p.coords[1], p.coords[2])

midx(p1::FlowPoint, p2::FlowPoint) = (p2.coords[1] + p1.coords[1]) / 2
# midx(ps::Vector{FlowPoint{T}}) where T<:Number = sum([p.coords[1] for p in ps])/length(ps)

midy(p1::FlowPoint, p2::FlowPoint) = (p2.coords[2] + p1.coords[2]) / 2
# midy(ps::Vector{FlowPoint{T}}) where T<:Number = sum([p.coords[2] for p in ps])/length(ps)

midpoint(p1::FlowPoint, p2::FlowPoint) = FlowPoint(midx(p1, p2), midy(p1, p2))
# midpoint(Ps::Vector{FlowPoint{T}}) where T<:Number = FlowPoint(midx(Ps), midy(Ps));

### doesn't seem to be needed
# function point2vec(p::FlowPoint2)
#     return [reim(p.x)...,reim(p.y)...]
# end;

## initialising the original integration domain

export make_init_points_rectangle
function make_init_points_rectangle(
    t1min::Real, t1max::Real,
    t2min::Real, t2max::Real,
    flow_bounds=[true, true, true, true],
    point_type=FlowPoint
)
    p1 = point_type(complex(t1min), complex(t2min))
    p1.active = flow_bounds[1]
    p2 = point_type(complex(t1min), complex(t2max))
    p2.active = flow_bounds[2]
    p3 = point_type(complex(t1max), complex(t2max))
    p3.active = flow_bounds[3]
    p4 = point_type(complex(t1max), complex(t2min))
    p4.active = flow_bounds[4]

    return [p1, p2, p3, p4]
end

export make_init_points_parallelogram
function make_init_points_parallelogram(
    timin::Real, timax::Real,
    ttmin::Real, ttmax::Real,
    flow_bounds=[true, true, true, true],
    point_type=FlowPoint
)
    p1 = point_type(complex(timin), complex(timin + ttmin))
    p1.active = flow_bounds[1]
    p2 = point_type(complex(timin), complex(timin + ttmax))
    p2.active = flow_bounds[2]
    p3 = point_type(complex(timax), complex(timax + ttmax))
    p3.active = flow_bounds[3]
    p4 = point_type(complex(timax), complex(timax + ttmin))
    p4.active = flow_bounds[4]

    return [p1, p2, p3, p4]
end


