module SaddlePoint

using Contour, GeometryBasics

using ..Types: Saddle, FlowPoint
using ..CriticalPoints: find_saddles_sobol
using ..LineIntersection: crosses_point, dissect_curve, intersection
using ..PathFlow: get_thimble, flow_up, find_intersection_point

export is_contributing
function is_contributing(ts_saddle::Saddle, S::Function, tmin::ComplexF64, tmax::ComplexF64;
    Ntimes::Int64=100)
    ts = ts_saddle[1]
    timags = range(imag(tmin), stop=imag(tmax), length=Ntimes)
    treals = range(real(tmin), stop=real(tmax), length=Ntimes)
    tlength = real(tmax - tmin)

    # Δi = timags[2]-timags[1]
    # Δr = treals[2]-treals[1]
    crossthresh = sqrt((step(timags))^2 + (step(treals))^2)

    Svals = [S(tr + im * ti) for tr in treals, ti in timags]

    real_axis = Contour.Curve2([(real(tmin) - tlength, 0.), (real(tmax) + tlength, 0.)])

    relevant = false
    S_saddle = S(ts)


    con_S_saddle_real = Contour.contour(collect(treals), collect(timags), real.(Svals), real.(S_saddle))
    for curve in con_S_saddle_real.lines
        saddle_ip = crosses_point(curve, GeometryBasics.Point(reim(ts)...), crossthresh)
        ### filter for those level lines that actually intersect the saddle point
        if !(saddle_ip == false)

            segs = dissect_curve(ts, curve, saddle_ip, crossthresh)
            #             saddle_ip = closest_intersection(saddle_ip, curve, GeometryBasics.Point(reim(ts)...), crossthresh)
            # #                 ### disect curve
            #             c1 = Contour.Curve2(curve.vertices[1:saddle_ip+1])
            #             c2 = Contour.Curve2(curve.vertices[saddle_ip+1:end])

            for c in segs
                actiondiff = S(complex(c.vertices[minimum([5, length(c.vertices)])]...)) - S(ts)
                if imag(-actiondiff) > 0 # ascent lines
                    if !isempty(intersection(c, real_axis))
                        relevant = true
                    end
                end
            end
        end
    end

    return relevant

end

export get_intersection_number!
function get_intersection_number!(
    S::Function, S_grad::Function, S_hessian::Function,
    saddle::Saddle, params::Dict
)::Nothing

    if isnothing(saddle.thimble)
        init_perturbation_radius = params["init_perturbation_radius"]
        max_iterations = params["max_iterations"]
        flow_step_factor = params["flow_step_factor"]
        subdivision_threshold = params["subdivision_threshold"]
        height_threshold = params["height_threshold"]
        gradient_normalisation_threshold = params["gradient_normalisation_threshold"]

        saddle.thimble = get_thimble(S, S_grad, S_hessian, saddle,
            init_perturbation_radius=init_perturbation_radius,
            max_iterations=max_iterations,
            flow_step_factor=flow_step_factor,
            subdivision_threshold=subdivision_threshold,
            gradient_normalisation_threshold=gradient_normalisation_threshold,
            height_threshold=height_threshold
        )
    end

    if isnothing(saddle.dual_thimble)
        flow_step_factor = params["flow_step_factor"]
        height_threshold = params["height_threshold"]
        max_iterations = params["max_iterations"]
        thimble, contributing = flow_up(S, S_grad, saddle.saddle, flow_step_factor, height_threshold, max_iterations)
        saddle.dual_thimble = thimble
    end

    intersection_point = find_intersection_point(saddle.dual_thimble)
    z_prime = conj(S_grad(intersection_point.coords[1]))
    determinant = imag(z_prime)
    saddle.intersection_number = sign(determinant)
    saddle.contributing = sign(determinant) != 0
    return nothing
end

export integrate_around_saddle_point
function integrate_around_saddle_point(ts_saddle::Saddle,
    S::Function, drv::Function, drv2::Function
    ; prefactor::Function=t -> 1.,
)

    ts = ts_saddle[1]
    S_ts = S(ts)

    int = prefactor(ts) * sqrt(-im * 2π / drv2(ts)) * exp(im * S_ts)
end

export integrate_SPM
function integrate_SPM(S::Function, drv::Function, drv2::Function,
    tmin::ComplexF64, tmax::ComplexF64
    ; prefactor::Function=t -> 1.)

    saddles = filter(ts -> real(tmin) < real(ts[1]) < real(tmax),
        find_saddles_sobol(drv, tmin, tmax, 300)
    )
    contributing_saddles = Saddle[]
    for ts in saddles
        if is_contributing(ts, S, tmin, tmax)
            integral = integrate_around_saddle_point(ts, S, drv, drv2, prefactor=prefactor)
            ts.integral = integral
            ts.contributing = true
            push!(contributing_saddles, ts)
        end
    end
    return contributing_saddles
end

end
