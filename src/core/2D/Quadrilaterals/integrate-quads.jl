module Integration

using ..Types: FlowPoint, Simplex

# QUADRILATERALS

#### integrate the quads
using FastGaussQuadrature

import Base.map
function map(p::Vector{Float64}, p1::FlowPoint, p2::FlowPoint, p3::FlowPoint, p4::FlowPoint)
    return ([p1[1], p1[2]] .* (1. - p[1]) * (1. - p[2]) +
            [p2[1], p2[2]] .* (1. + p[1]) * (1. - p[2]) +
            [p3[1], p3[2]] .* (1. + p[1]) * (1. + p[2]) +
            [p4[1], p4[2]] .* (1. - p[1]) * (1. + p[2])) / 4.
end


function jacobian(p::Vector{Float64}, p1::FlowPoint, p2::FlowPoint, p3::FlowPoint, p4::FlowPoint)
    A = +(p1[1] - p3[1]) * (p2[2] - p4[2]) - (p1[2] - p3[2]) * (p2[1] - p4[1])
    B = -(p1[1] - p2[1]) * (p3[2] - p4[2]) + (p1[2] - p2[2]) * (p3[1] - p4[1])
    C = +(p2[1] - p3[1]) * (p1[2] - p4[2]) - (p2[2] - p3[2]) * (p1[1] - p4[1])
    return (A + B * p[1] + C * p[2]) / 8.
end

export integrate_quadrilateral
function integrate_quadrilateral(
    f::Function,
    quad::Simplex{4,FlowPoint}, n::Int64=7;
    prefactor::Function=t -> ones(2)
)

    p1, p2, p3, p4 = quad.vertices

    x, w = gausslegendre(n)
    y = x
    sum = zero(prefactor([p1[1], p1[2]])) .+ 0im
    for i = 1:n, j = 1:n
        jac = -jacobian([x[i], y[j]], p1, p2, p3, p4) # this minus sign here comes from that debugging experiment in the 2024-10-20 figures spectra... NB

        ti, tr = map([x[i], x[j]], p1, p2, p3, p4)
        action = f([ti, tr])

        sum = sum + jac * prefactor([ti, tr]) * exp(action) * w[i] * w[j]
    end

    return sum

end

end