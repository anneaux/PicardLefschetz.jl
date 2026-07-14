using StaticArrays
using OrdinaryDiffEq

module GradientDescent

# Calculates the vector field along which the flow is evaluated. 
function flow_vector_field(w::AbstractVector{T}, S_gradient::Function, direction::Symbol)::Vector{T} where {T<:Real}
    D = length(w) ÷ 2
    x = @views w[1:D]
    y = @views w[D+1:end]
    z = SVector{D,Complex{T}}(x[j] + im * y[j] for j in 1:D)
    gradient = S_gradient(z)
    dw = Vector{T}(undef, 2 * D)
    sign_val = (direction == :descent) ? -1.0 : 1.0

    for j in 1:D
        dw[j] = sign_val * real(gradient[j])
        dw[D+j] = -sign_val * imag(gradient[j])
    end

    return dw
end

# Calculates the analytical expression for the Jacobian in the vector field on the thimble.
function flow_jacobian(w::AbstractVector{T}, S_hessian::Function, direction::Symbol)::Matrix{T} where {T<:Real}
    D = length(w) ÷ 2
    x = @views w[1:D]
    y = @views w[D+1:end]
    z = SVector{D,Complex{T}}(x[j] + im * y[j] for j in 1:D)

    H = S_hessian(z)

    J = Matrix{T}(undef, 2 * D, 2 * D)

    sigma = (direction == :descent) ? -1.0 : 1.0

    for k in 1:D
        for j in 1:D
            A_jk = real(H[j, k])
            B_jk = imag(H[j, k])

            J[j, k] = sigma * A_jk
            J[j, D+k] = -sigma * B_jk
            J[D+j, k] = -sigma * B_jk
            J[D+j, D+k] = -sigma * A_jk
        end
    end
    return J
end

# Flows the points from the starting domain along the thimble.
export flow_points!
"""
Flows the points from the starting domain along the thimble. This is a parallelised version, which flows points simultaneously. 

@param points::Vector{Point{D}} The initial points for the flow. Note that these points are the initial positions in the space.
@param S::Function The integrand action, S(z).
@param S_gradient::Function The gradient of the integrand action, S(z).
@param S_hessian::Function The Hessian of the integrand action, S(z).
@param direction::Symbol The sign of the evolution, where negative implies descent and positive implies ascent.
@param h_threshold::Real The threshold value for the action. If the imaginary component of the action is less than this value, the flow is stopped.
@param steps::Int The number of steps to take for the gradient flow.
@param δ_init::Real The initial step size.

@return Nothing. Modifies the points vector in place.
"""
function flow_points!(
    points::Vector{Point{D}},
    S::Function,
    S_gradient::Function,
    S_hessian::Function,
    direction::Symbol,
    h_threshold::Real,
    steps::Int,
    δ_init::Real)::Nothing where D

    Threads.@threads for i in 1:length(points)
        if points[i].active
            points[i].coords, active = flow_single_point(points[i].coords, S, S_gradient, S_hessian, direction, h_threshold, steps, δ_init)
            points[i].active = active
        end
    end
end

# Wrapper for the vector field calculation.
function flow_ode_vector_field!(
    dw::AbstractMatrix{T},
    w::AbstractVector{T},
    p::Tuple{F1,Symbol,F2,F3,Real},
    t::Real)::Nothing where {T<:Real,F1,F2,F3}

    S_gradient = p[1]
    direction = p[2]
    dw .= flow_vector_field(w, S_gradient, direction)
    return nothing
end

# Wrapper for the Jacobian calculation.
function flow_ode_jacobian!(
    J::AbstractMatrix{T},
    w::AbstractVector{T},
    p::Tuple{F1,Symbol,F2,F3,Real},
    t::Real)::Nothing where {T<:Real,F1,F2,F3}

    S_hessian = p[3]
    direction = p[2]
    J .= flow_jacobian(w, S_hessian, direction)
    return nothing
end

# Defines the cutoff condition for the gradient flow, i.e. enforces the h_threshold.
function cutoff_condition(
    w::AbstractVector{T},
    t::Real,
    integrator)::Bool where {T<:Real}

    direction = integrator.p[2]
    S = integrator.p[4]
    h_threshold = integrator.p[5]

    D = length(w) ÷ 2
    z = SVector{D,Complex{T}}(w[j] + im * w[D+j] for j in 1:D)

    if direction == :descent
        return real(S(z)) <= h_threshold
    else
        return false
    end
end

# Defines the action of the cutoff condition being met.
function cutoff_affect!(integrator)::Nothing
    terminate!(integrator)
    return nothing
end

# Defines the flow for a single point.
function flow_single_point(
    initial_point::Point{D},
    S::Function,
    S_gradient::Function,
    S_hessian::Function,
    direction::Symbol,
    h_threshold::Real,
    steps::Int,
    δ_init::Real)::Tuple{Point{D},Bool} where D

    # Converting the coordinates from a complex vector with dim = D, to a real vector with dim = 2D
    w0 = Vector{Float64}(undef, 2 * D)
    for j in 1:D
        w0[j] = real(initial_point.coords[j])
        w0[D+j] = imag(initial_point.coords[j])
    end

    # Packing the parameters and the functions required for the evaluation of the ODE.
    # Defining the components of the ODE problem, i.e. framing it for the library.
    p = (S_gradient, direction, S_hessian, S, h_threshold)
    ode_function = ODEFunction(flow_ode_vector_field!; jac=flow_ode_jacobian!)
    timespan = (0.0, Float64(steps))
    callback = DiscreteCallback(cutoff_condition, cutoff_affect!)
    problem = ODEProblem(ode_function, w0, timespan, p)
    solution = solve(
        problem,
        Rodas5P(),
        dt=δ_init,
        adaptive=true,
        reltol=1e-8,
        abstol=1e-8,
        callback=callback,
        maxiters=steps
    )

    w_final = solution.u[end]
    coords_final = SVector{D,ComplexF64}(w_final[j] + im * w_final[D+j] for j in 1:D)
    active = real(S(coords_final)) > h_threshold

    return coords_final, active
end

end