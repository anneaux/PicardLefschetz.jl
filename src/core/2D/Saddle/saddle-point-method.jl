module SaddlePoint

using ..Types: Saddle
import Contour
using StaticArrays
using ForwardDiff
using LinearAlgebra

include("saddles-contributing.jl")
include("saddles-generic.jl")

gradient_vector_field::Union{Nothing,Function} = nothing

function hessian_root(h::AbstractArray)
    if size(h) == (2, 2)
        determinant = h[1, 1] * h[2, 2] - h[1, 2] * h[2, 1]
        return im * 2 * π / sqrt(determinant)
    end
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
function get_intersection_number!(S::Function, saddle::Saddle, params::Dict)::Nothing
    flowstepfactor = Float64(params["flow_step_factor"])
    
    # Compute the velocity field.
    SaddlePoint.gradient_vector_field = velocity_field(S)

    # Get the positive Hessian eigenvectors
    X = @SVector[real(saddle.saddle.coords[1]), imag(saddle.saddle.coords[1]), real(saddle.saddle.coords[2]), imag(saddle.saddle.coords[2])]
    eigenvectors = get_positive_hessian_eigenvectors(S, X)

    # Initial orientation O = 1 because the necklace is constructed with a fixed orientation
    O = 1.0

    # Get the intersection simplex from the necklace
    mesh = convert_to_mesh(saddle.dual_thimble_boundary)
    intersection_simplex = find_intersection_for_contribution(mesh, saddle, flowstepfactor=flowstepfactor)

    # Construct the geometric tangent vectors at the intersection simplex
    # P1 and P2 are the vertices of the simplex
    P1 = intersection_simplex.vertices[1].coords
    P2 = intersection_simplex.vertices[2].coords
    
    v_necklace = [real(P2[1]) - real(P1[1]), imag(P2[1]) - imag(P1[1]), real(P2[2]) - real(P1[2]), imag(P2[2]) - imag(P1[2])]
    
    midpoint_z1 = (P1[1] + P2[1]) / 2.0
    midpoint_z2 = (P1[2] + P2[2]) / 2.0
    midpoint_X = @SVector[real(midpoint_z1), imag(midpoint_z1), real(midpoint_z2), imag(midpoint_z2)]
    
    v_flow = gradient_vector_field(midpoint_X)

    # U is the basis [v_flow, v_necklace]
    # We want det(M) where M is the restriction to the imaginary coordinates (the 2nd and 4th components)
    U21, U22 = v_flow[2], v_necklace[2]
    U41, U42 = v_flow[4], v_necklace[4]
    
    det_M = U21 * U42 - U22 * U41
    saddle.intersection_number = Int(sign(det_M) * sign(O))
    
    return nothing
end

function get_positive_hessian_eigenvectors(S::Function, X::SVector{4,Real})
    imag_S(X) = imag(S(@SVector[X[1] + im * X[2], X[3] + im * X[4]]))
    H = ForwardDiff.hessian(imag_S, X)
    eigen_decomposition = eigen(H)
    positive_indices = findall(λ -> λ > 0, eigen_decomposition.values)
    return eigen_decomposition.vectors[:, positive_indices]
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