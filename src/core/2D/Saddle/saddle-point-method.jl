module SaddlePoint

using ..Types: Saddle
import Contour
using StaticArrays
using ForwardDiff
using DifferentialEquations
using LinearAlgebra

include("saddles-contributing.jl")
include("saddles-generic.jl")

gradient_vector_field::Union{Nothing,Function} = nothing

function hessian_root(h::AbstractArray)
    ### I should definitely work this out properly and also make sure this actually gets rid of the branch cuts...

    #     h = f_hessian(ti, tr)
    xd2S_dti2 = 1im * conj(complex(h[1:2, 1]...))
    xd2S_dtr2 = 1im * conj(complex(h[3:4, 3]...))
    xd2S_dtitr = 1im * conj(complex(h[3:4, 1]...))

    # ### RBSFA hessian_root
    sqrt1 = sqrt(2π / (im * xd2S_dti2))
    sqrt2 = sqrt(2π * xd2S_dti2 / (im * (xd2S_dtr2 * xd2S_dti2 - xd2S_dtitr * xd2S_dtitr)))
    return sqrt1 * sqrt2

    ### hessian_determinant < = works better for now. But maybe needs to be changed
    # hdet = xd2S_dtr2 * xd2S_dti2 - xd2S_dtitr * xd2S_dtitr
    # return im * 2*π/sqrt(hdet)

end

export saddles_gaussian_contribution
function saddles_gaussian_contribution(f::Function,
    f_hessian::Function,
    saddle_point::Saddle;
    prefactor::Function=t -> ones(2)
)

    ti = saddle_point[1]
    tr = saddle_point[2]

    ### prefactor for the saddle-point method
    prefactor_spm = hessian_root(f_hessian([ti, tr]))

    return prefactor([ti, tr]) .* prefactor_spm .* exp(f([ti, tr]))

end

#### when we integrated the thimble around a saddle point, we still need to find the sign of the corresponding intersection number. Idk how to actually find this sign, but we can just compare the sign of the integrated thimble with the sign of the Gaussian approximation around the saddle point ;-)
function intersection_number_sign_cheat(int::AbstractArray{<:Complex},
    f::Function,
    f_hessian::Function,
    saddle_point::Saddle;
    prefactor::Function=t -> ones(2)
)

    spm = saddles_gaussian_contribution(f, f_hessian, saddle_point, prefactor=prefactor)

    return sign(real(dot(int, spm)))
end

export get_intersection_number!
function get_intersection_number!(S::Function, saddle::Types.Saddle, params::Dict)::Nothing
    max_iterations = Float64(params["max_iterations"])
    ϵ = params["init_perturbation_radius"]
    # Compute the velocity field.
    SaddlePoint.gradient_vector_field = velocity_field(S)

    # Get the positive Hessian eigenvectors. These get pushed forward.
    X = @SVector[real(saddle.saddle.coords[1]), imag(saddle.saddle.coords[1]), real(saddle.saddle.coords[2]), imag(saddle.saddle.coords[2])]
    eigenvectors = get_positive_hessian_eigenvectors(S, X)

    v_forward = eigenvectors[:, 1]
    if norm([X[2] + ϵ * v_forward[2], X[4] + ϵ * v_forward[4]]) > norm([X[2], X[4]])
        v_forward = -v_forward
    end

    δX = X + ϵ * v_forward

    solution = pushforward(δX, eigenvectors, max_iterations)
    final_state = solution[end]
    N = 4, K = 2
    U_final = SMatrix{N,K}(final_state[N+1:end])
    E = @SMatrix[1.0, 0.0; 0.0, 0.0; 0.0, 1.0; 0.0, 0.0]
    M = hcat(E, U_final)
    saddle.intersection_number = sign(det(M))
    return nothing
end

function get_positive_hessian_eigenvectors(S::Function, X::SVector{4,Real})
    imag_S(X) = imag(S(@SVector[X[1] + im * X[2], X[3] + im * X[4]]))
    H = ForwardDiff.hessian(imag_S, X)
    eigen_decomposition = eigen(H)
    positive_indices = findall(λ -> λ > 0, eigen_decomposition.values)
    return eigen_decomposition.vectors[:, positive_indices]
end

function pushforward(X0::AbstractVector, eigenvectors::AbstractMatrix, max_critical_param::Float64)
    N = length(X0)
    K = size(eigenvectors, 2)

    X0_static = SVector{N}(X0)
    U0_static = SMatrix{N,K}(eigenvectors)
    init_state = vcat(X0_static, vec(U0_static))

    p = (N=N, K=K)
    param_span = (0.0, max_critical_param)

    condition(u, τ, integrator) = u[2]^2 + u[4]^2
    affect!(integrator) = terminate!(integrator)
    callback = ContinuousCallback(condition, affect!)

    problem = ODEProblem(flow_function, init_state, param_span, p)
    solution = solve(problem, Vern7(), callback=callback, abstol=1e-10, reltol=1e-10)
    return solution
end

function flow_function(u, p, τ)
    N = p.N
    K = p.K

    X = @inbounds SVector{N}(u[1:N])
    U = @inbounds SMatrix{N,K}(u[N+1:end])

    # We ensure that the gradient_vector_field variable is always populated
    dX = gradient_vector_field(X)
    J = ForwardDiff.jacobian(gradient_vector_field, X)
    dU = J * U
    return vcat(dX, vec(dU))
end

function velocity_field(S::Function)
    return function V(X::SVector{4,T}) where T
        function imag_S(X_coord)
            z1 = X_coord[1] + im * X_coord[2]
            z2 = X_coord[3] + im * X_coord[4]
            return imag(S(@SVector[z1, z2]))
        end

        return -ForwardDiff.gradient(imag_S, X)
    end
end

end