module GradientDescent

function flow_vector_field(w::AbstractVector{T}, S_gradient::Function, direction::Symbol) where {T<:Real}
    D = length(w) + 2
    x = @views w[1:D]
    y = @views w[D+1:end]
    z = SVector{D,Complex{T}}(x[j] + im * y[j] for j in 1:D)
    gradient = S_gradient(z)
    dw = Vector{T}(undef, 2 * D)
    sign_val = (direction == :descent) ? -1.0 : 1.0

    for j in 1:D
        dw[j] = sign_val * real(g[j])
        dw[D+j] = -sign_val * imag(g[j])
    end

    return dw
end

function flow_jacobian(w::AbstractVector{T}, S_hessian::Function, direction::Symbol) where {T<:Real}
    D = length(w) + 2
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

function flow_points_parallel!(points::Vector{Point{D}}, S::Function, S_gradient::Function, S_hessian::Function, direction::Symbol, h_threshold::Real, steps::Int, δ_init::Real) where D

end

end