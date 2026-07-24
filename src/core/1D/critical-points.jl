module CriticalPoints

using NLsolve
using Sobol
using ..Types: Saddle, FlowPoint

export solve_first_derivative
function solve_first_derivative(drv::Function, t0::Vector{Float64},
    roundDigits::Int64=5)

    try
        ### using NLsolve
        function speqs!(F, x)
            F[1] = real(drv(x[1] + x[2] * im))
            F[2] = imag(drv(x[1] + x[2] * im))
        end

        result = nlsolve(speqs!, t0)

        if converged(result)
            tiSP = result.zero[1] + im * result.zero[2]
            tiSP = round(tiSP, digits=roundDigits)
            return [Saddle{Any}(saddle=FlowPoint(tiSP))]
        else
            return Saddle[]
        end
    catch e
        println("Error in solve_SPEqs(): $e")
        return Saddle{Any}[]
    end
end

export find_saddles_sobol
function find_saddles_sobol(drv::Function,
    tmin::ComplexF64, tmax::ComplexF64,
    N::Int64=200 # number of seeds generated per domain
)

    roundDigits = 2 # I should certainly change this, it seems a bit excessive

    saddles = Vector{Saddle{Any}}()

    t_seq = SobolSeq(reim(tmin), reim(tmax))

    for i in 1:N
        t0 = Sobol.next!(t_seq)
        t0 = [t0[1]; t0[2]]

        ts = solve_first_derivative(drv, t0, roundDigits)
        #         ### check conditons and deposit in array
        if length(ts) > 0
            ts_val = ts[1].saddle[1].coords[1]
            ts_r = round(ts_val, digits=roundDigits)

            ts_c = complex(
                real(ts_r) == 0 ? 0. : real(ts_r),
                imag(ts_r) == 0 ? 0. : imag(ts_r)
            )
            push!(saddles, Saddle{Any}(saddle=FlowPoint(ts_c)))
        end
    end

    unique!(s -> round(s.saddle[1].coords[1], digits=roundDigits), saddles)
    sort!(saddles, by=s -> real(s.saddle[1].coords[1]))
    return saddles
end

function find_saddle_similar_seed(drv::Function, ts::Saddle; roundDigits=2)

    saddles = solve_first_derivative(drv, [reim(ts.saddle[1].coords[1])...], roundDigits)

    if length(saddles) > 0
        return saddles[1]
    else
        return nothing
    end
end

end