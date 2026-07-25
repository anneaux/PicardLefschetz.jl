module Integration

using FastGaussQuadrature
using LinearAlgebra
using ..PathFlow: flow_up, initialise, flow_down!, subdivide_rep

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

export integrate_thimble
function integrate_thimble(S::Function,
    points::Vector, simplices::Vector;
    prefactor::Function=t -> 1.)
    points_r = map(pp -> pp.coords[1], points)
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
export integrate_thimble
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

        #filter!(sim->sim.active, simplices)
    end

    return int, points, simplices
end

end
