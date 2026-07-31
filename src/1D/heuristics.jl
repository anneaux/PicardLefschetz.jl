# --------------------------------------------------------------------------
# Preset heuristic parameters.
# --------------------------------------------------------------------------

const PRESET_FAST = [0.36964, 0.27705, 0.01121, 0.20152, 50.0, -789.24]
const PRESET_ACCURATE = [1.81854, 0.14597, 0.01257, 0.06316, 213.0, -1050.68]
const PRESET_THIMBLE = [1.73072, 0.20915, 0.03509, 2.26757, 966.0, -1995.76]

# Return preset parameter vector.
function get_preset(preset::Symbol)
    if preset == :fast
        return PRESET_FAST
    elseif preset == :accurate
        return PRESET_ACCURATE
    elseif preset == :thimble
        return PRESET_THIMBLE
    else
        throw(ArgumentError("Unknown preset: $preset. Choose :fast, :accurate, or :thimble"))
    end
end


# --------------------------------------------------------------------------
# Parameter utilities.
# --------------------------------------------------------------------------

# Unpack heuristic parameter vector.
function unpack_params(p::AbstractVector{<:Real})
    if length(p) == 5
        α_init, α_subdiv, α_grad_scale, α_grad, Nflow = p
        h_threshold = -300.0
    elseif length(p) == 6
        α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = p
    else
        throw(ArgumentError("Unsupported parameter vector length $(length(p))"))
    end

    return α_init, α_subdiv, α_grad_scale, α_grad, Int(round(Nflow)), h_threshold
end

# Resolve parameters from a preset symbol, parameter vector, or existing params struct/NamedTuple.
function resolve_heuristics(preset_or_params=PRESET_ACCURATE; kwargs...)
    base_params = if preset_or_params isa Symbol
        p_vec = get_preset(preset_or_params)
        α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = unpack_params(p_vec)
        (
            α_init=α_init,
            α_subdiv=α_subdiv,
            α_grad_scale=α_grad_scale,
            α_grad=α_grad,
            Nflow=Nflow,
            h_threshold=h_threshold
        )
    elseif preset_or_params isa AbstractVector{<:Real}
        α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = unpack_params(preset_or_params)
        (
            α_init=α_init,
            α_subdiv=α_subdiv,
            α_grad_scale=α_grad_scale,
            α_grad=α_grad,
            Nflow=Nflow,
            h_threshold=h_threshold
        )
    elseif preset_or_params isa NamedTuple
        preset_or_params
    else
        throw(ArgumentError("Unsupported preset or params type: $(typeof(preset_or_params))"))
    end

    return isempty(kwargs) ? base_params : merge(base_params, NamedTuple(kwargs))
end

# Evaluate S on scalar or vector arguments.
_eval_S(S::Function, p::Number) = S(p)
_eval_S(S::Function, p::AbstractVector) = S(p...)
_eval_S(S::Function, p::Tuple) = S(p...)


# --------------------------------------------------------------------------
# Oscillation radius.
# --------------------------------------------------------------------------

# Find the first oscillatory boundary along a ray.
function ballradius(S::Function, ξ, Cω, d_norm; r_max::Float64=10.0, no_pts::Int=101)
    ray(r) = ξ .+ r .* d_norm
    S_ξ = _eval_S(S, ξ)
    un(r) = abs(_eval_S(S, ray(r)) - S_ξ)^2 - Cω^2

    r_vals = range(0.0, r_max, length=no_pts)
    u_vals = [un(r) for r in r_vals]

    for i in 1:(length(r_vals)-1)
        if (u_vals[i] <= 0.0 && u_vals[i+1] >= 0.0) || (u_vals[i] >= 0.0 && u_vals[i+1] <= 0.0)
            r_left, r_right = r_vals[i], r_vals[i+1]
            for _ in 1:15
                r_mid = (r_left + r_right) / 2.0
                if (un(r_mid) <= 0.0) == (u_vals[i] <= 0.0)
                    r_left = r_mid
                else
                    r_right = r_mid
                end
            end
            return (r_left + r_right) / 2.0
        end
    end
    error("Oscillatory boundary not found before r_max.")
end

# Uses a finite-difference Hessian to estimate local flow directions.
function get_hessian_eigenvectors!(directions::Vector{ComplexF64}, ξ::ComplexF64, S::Function, sign::Symbol)::Nothing
    X = [real(ξ), imag(ξ)]
    hessian = FiniteDiff.finite_difference_hessian(tvec -> imag(S(complex(tvec[1], tvec[2]))), X)
    eigen_decomposition = eigen(hessian)
    eigenvalues = eigen_decomposition.values
    eigenvectors = eigen_decomposition.vectors
    for i in eachindex(eigenvalues)
        if sign == :descent
            if eigenvalues[i] > 0
                v = complex(eigenvectors[:, i][1], eigenvectors[:, i][2])
                push!(directions, v)
                push!(directions, -v)
            end
        else
            if eigenvalues[i] < 0
                v = complex(eigenvectors[:, i][1], eigenvectors[:, i][2])
                push!(directions, v)
                push!(directions, -v)
            end
        end
    end
    return nothing
end

# Compute directional oscillation radii.
function get_directional_r_osc(S::Function, ξ::ComplexF64, ω::Float64; Cball=2π, r_max=10.0)
    directions = ComplexF64[]
    get_hessian_eigenvectors!(directions, ξ, S, :descent)
    get_hessian_eigenvectors!(directions, ξ, S, :ascent)

    if isempty(directions)
        directions = [1.0 + 0.0im, -1.0 + 0.0im]
    end

    pairs = Tuple{ComplexF64,Float64}[]
    for d in directions
        d_norm = d / norm(d)
        r = ballradius(S, ξ, Cball / ω, d_norm; r_max=r_max)
        push!(pairs, (d_norm, r))
    end
    return pairs
end

# Compute directional oscillation radii.
function get_directional_r_osc(S::Function, f_hessian::Function, ξ::AbstractVector{ComplexF64}, ω::Float64; Cball=2π, r_max=10.0)
    hessian = f_hessian(ξ...)
    eigvec_mat = eigvecs(hessian)
    eigenvectors = [[complex(vec[1:2]...), complex(vec[3:4]...)] for vec in eachcol(eigvec_mat)]

    pairs = Tuple{Vector{ComplexF64},Float64}[]
    for d in eigenvectors
        d_norm = d / norm(d)
        r = ballradius(S, ξ, Cball / ω, d_norm; r_max=r_max)
        push!(pairs, (d_norm, r))
    end
    return pairs
end


# --------------------------------------------------------------------------
# Main heuristic scaling routines.
# --------------------------------------------------------------------------

# Scale heuristics from the oscillation radius.
function get_pl_heuristics_1d(
    S::Function,
    drv::Function,
    ξ::Union{ComplexF64,Nothing}=nothing,
    ω::Float64=1.0,
    p=PRESET_ACCURATE;
    tmin::Real=-10.0,
    tmax::Real=10.0,
    Cball=2π,
    r_max=10.0,
    kwargs...
)
    resolved = resolve_heuristics(p; kwargs...)
    α_init = resolved.α_init
    α_subdiv = resolved.α_subdiv
    α_grad_scale = resolved.α_grad_scale
    α_grad = resolved.α_grad
    Nflow = resolved.Nflow
    h_threshold = resolved.h_threshold

    ξ_eff = if ξ !== nothing
        ξ
    else
        saddles = find_saddles_sobol(drv, ComplexF64(tmin), ComplexF64(tmax))
        !isempty(saddles) ? saddles[1] : complex(Float64(tmin + tmax) / 2.0, 0.0)
    end

    directional_radii =
        get_directional_r_osc(S, ξ_eff, ω; Cball=Cball, r_max=r_max)

    r_osc = minimum(radius for (_, radius) in directional_radii)

    Δinit = α_init * r_osc
    subdividethreshold = α_subdiv * r_osc
    grad_scale_radius = α_grad_scale * r_osc

    gradnthreshold = α_grad * maximum(
        norm(drv(ξ_eff + grad_scale_radius * direction))
        for (direction, _) in directional_radii
    )

    computed = (
        Δinit=Δinit,
        subdividethreshold=subdividethreshold,
        grad_scale_radius=grad_scale_radius,
        gradnthreshold=gradnthreshold,
        h_threshold=h_threshold,
        r_osc=r_osc,
        directional_radii=directional_radii,
        Nflow=Nflow,
        flowstepfactor=0.015
    )
    return isempty(kwargs) ? computed : merge(computed, NamedTuple(kwargs))
end

# Scale heuristics from the oscillation radius.
function get_pl_heuristics_2d(
    S::Function,
    f_grad::Function,
    f_hessian::Function,
    ξ::AbstractVector{ComplexF64},
    ω::Float64,
    p=PRESET_ACCURATE;
    Cball=2π,
    r_max=50.0,
    kwargs...
)
    resolved = resolve_heuristics(p; kwargs...)
    α_init = resolved.α_init
    α_subdiv = resolved.α_subdiv
    α_grad_scale = resolved.α_grad_scale
    α_grad = resolved.α_grad
    Nflow = resolved.Nflow
    h_threshold = resolved.h_threshold

    directional_radii =
        get_directional_r_osc(
            S,
            f_hessian,
            ξ,
            ω;
            Cball=Cball,
            r_max=r_max
        )

    r_osc = minimum(radius for (_, radius) in directional_radii)

    eigvecfactorinit = α_grad_scale * r_osc
    subdividethreshold = α_subdiv * r_osc
    flowstepfactor = α_init * r_osc
    Δinit = α_init * r_osc

    gradnthreshold = α_grad * maximum(
        norm(f_grad((ξ .+ eigvecfactorinit .* direction)...))
        for (direction, _) in directional_radii
    )

    Ninit = max(4, Int(round(2π * eigvecfactorinit / flowstepfactor)))

    computed = (
        eigvecfactorinit=eigvecfactorinit,
        subdividethreshold=subdividethreshold,
        flowstepfactor=flowstepfactor,
        gradnthreshold=gradnthreshold,
        h_threshold=h_threshold,
        r_osc=r_osc,
        directional_radii=directional_radii,
        Nflow=Nflow,
        Ninit=Ninit,
        Δinit=Δinit
    )
    return isempty(kwargs) ? computed : merge(computed, NamedTuple(kwargs))
end