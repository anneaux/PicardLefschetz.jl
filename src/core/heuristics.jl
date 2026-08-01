using ..Types: RealDomain
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

export resolve_heuristics
"""
    resolve_heuristics(S, S_grad, saddle_points, action_scaling_factor, preset; integration_domain, kwargs...)

Resolves the heuristic parameters for the 1D Picard-Lefschetz method from the action function, its gradient, saddle points, action scaling factor, preset, and integration domain.

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The first derivative of the action function.
- `saddle_points::Union{ComplexF64,Nothing}`: The saddle points to use for the heuristic.
- `action_scaling_factor::Float64`: The action scaling factor, e.g. \$\\hbar\$ in physics.
- `preset::Symbol`: The preset to use for the heuristic.
- `integration_domain::RealDomain`: The integration domain.
- `kwargs...`: User-valued parameters. These will overwrite the heuristic parameters.

# Returns
- `NamedTuple`: The heuristic parameters.
"""
function resolve_heuristics(
    S::Function,
    S_grad::Function,
    saddle_points::Union{ComplexF64,Nothing}=nothing,
    action_scaling_factor::Float64=1.0,
    preset=PRESET_ACCURATE;
    integration_domain::RealDomain,
    kwargs...
)
    computed = get_pl_heuristics_1d(S, S_grad, saddle_points, action_scaling_factor, preset; integration_domain=integration_domain)
    return isempty(kwargs) ? computed : merge(computed, NamedTuple(kwargs))
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

# Compute directional oscillation radii.
function get_directional_r_osc(S::Function, ξ::ComplexF64, ω::Float64; Cball=2π, r_max=10.0)
    directions = ComplexF64[]
    Methods1D.PathFlow.get_hessian_eigenvectors!(directions, ξ, S, :descent)
    Methods1D.PathFlow.get_hessian_eigenvectors!(directions, ξ, S, :ascent)

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
export get_pl_heuristics_1d
"""
    get_pl_heuristics_1d(S, S_grad, saddle_points, action_scaling_factor, preset; integration_domain, periodicity, search_radius)

Calculates the parameters to be used in the 1D Picard-Lefschetz method from the action function, its gradient, saddle points, action scaling factor, preset, integration domain, periodicity, and search radius.

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The gradient of the action function.
- `saddle_points::Union{Nothing,<:AbstractVector{ComplexF64}}=nothing`: The saddle points to use. If `nothing`, the saddle points will be found numerically.
- `action_scaling_factor::Float64=1.0`: The action scaling factor.
- `preset::Symbol=PRESET_ACCURATE`: The preset to use for the heuristic.
- `integration_domain::RealDomain`: The integration domain.
- `periodicity::Float64=2π`: The periodicity of the action function.
- `search_radius::Float64=10.0`: The radius of the search for saddle points.

# Returns
- `NamedTuple`: The heuristic parameters.
"""
function get_pl_heuristics_1d(
    S::Function,
    S_grad::Function,
    saddle_points::Union{Nothing,<:AbstractVector{ComplexF64}}=nothing,
    action_scaling_factor::Float64=1.0,
    preset=PRESET_ACCURATE;
    integration_domain::RealDomain,
    periodicity=2π,
    search_radius=10.0
)

    α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = unpack_params(get_preset(preset; kwargs...))

    ξ_eff = if saddle_points !== nothing
        saddle_points
    else
        saddles = find_saddles_sobol(S_grad, integration_domain.min, integration_domain.max)
        !isempty(saddles) ? saddles[1] : complex(Float64(integration_domain.min + integration_domain.max) / 2.0, 0.0)
    end

    directional_radii =
        get_directional_r_osc(S, ξ_eff, action_scaling_factor; Cball=periodicity, r_max=search_radius)

    r_osc = minimum(radius for (_, radius) in directional_radii)

    Δinit = α_init * r_osc
    subdividethreshold = α_subdiv * r_osc
    grad_scale_radius = α_grad_scale * r_osc

    gradnthreshold = α_grad * maximum(
        norm(S_grad(ξ_eff + grad_scale_radius * direction))
        for (direction, _) in directional_radii
    )

    computed = (
        init_point_count=Δinit,
        subdivision_threshold=subdividethreshold,
        gradient_normalisation_threshold=gradnthreshold,
        height_threshold=h_threshold,
        oscillation_radius=r_osc,
        directional_radii=directional_radii,
        max_iterations=Nflow,
        flow_step_factor=0.015
    )
    return isempty(kwargs) ? computed : merge(computed, NamedTuple(kwargs))
end

# Scale heuristics from the oscillation radius.
export get_pl_heuristics_2d
"""
    get_pl_heuristics_2d(S, S_grad, S_hessian, saddle_points, action_scaling_factor, preset; periodicity, search_radius)

Calculates the parameters to be used in the 2D Picard-Lefschetz method from the action function, its gradient, saddle points, action scaling factor, preset, integration domain, periodicity, and search radius.

# Arguments
- `S::Function`: The action function.
- `S_grad::Function`: The gradient of the action function.
- `S_hessian::Function`: The Hessian of the action function.
- `saddle_points::Union{Nothing,<:AbstractVector{ComplexF64}}=nothing`: The saddle points to use. If `nothing`, the saddle points will be found numerically.
- `action_scaling_factor::Float64=1.0`: The action scaling factor.
- `preset::Symbol=PRESET_ACCURATE`: The preset to use for the heuristic.
- `periodicity::Float64=2π`: The periodicity of the action function.
- `search_radius::Float64=10.0`: The radius of the search for saddle points.

# Returns
- `NamedTuple`: The heuristic parameters.
"""
function get_pl_heuristics_2d(
    S::Function,
    S_grad::Function,
    S_hessian::Function,
    saddle_points::Union{Nothing,<:AbstractVector{ComplexF64}},
    action_scaling_factor::Float64,
    preset=PRESET_ACCURATE;
    periodicity=2π,
    search_radius=50.0
)

    α_init, α_subdiv, α_grad_scale, α_grad, Nflow, h_threshold = unpack_params(get_preset(preset; kwargs...))

    directional_radii =
        get_directional_r_osc(
            S,
            S_hessian,
            saddle_points,
            action_scaling_factor;
            Cball=periodicity,
            r_max=search_radius
        )

    r_osc = minimum(radius for (_, radius) in directional_radii)

    eigvecfactorinit = α_grad_scale * r_osc
    subdividethreshold = α_subdiv * r_osc
    flowstepfactor = α_init * r_osc

    gradnthreshold = α_grad * maximum(
        norm(S_grad((ξ .+ eigvecfactorinit .* direction)...))
        for (direction, _) in directional_radii
    )

    Ninit = max(4, Int(round(2π * eigvecfactorinit / flowstepfactor)))

    computed = (
        init_perturbation_radius=eigvecfactorinit,
        subdivision_threshold=subdividethreshold,
        flow_step_factor=flowstepfactor,
        gradient_normalisation_threshold=gradnthreshold,
        height_threshold=h_threshold,
        oscillation_radius=r_osc,
        directional_radii=directional_radii,
        max_iterations=Nflow,
        init_point_count=Ninit
    )
    return isempty(kwargs) ? computed : merge(computed, NamedTuple(kwargs))
end