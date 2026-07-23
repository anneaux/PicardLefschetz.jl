module SaddlePoint

using ..Types: Saddle
import Contour

include("saddles-contributing.jl")
include("saddles-generic.jl")

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

    ti = saddle_point.saddle[1].coords[1]
    tr = saddle_point.saddle[2].coords[1]

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

end