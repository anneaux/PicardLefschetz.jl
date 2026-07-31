using GLMakie
using LinearAlgebra
using GeometryBasics
using LaTeXStrings

export run_and_plot, record_thimble_flow_animation, plot_thimble_3d


# --------------------------------------------------------------------------
# 2D Thimble plotting.
# --------------------------------------------------------------------------

# Plot a 1D complex thimble contour with background action contours.
function run_and_plot(ax, S::Function, points, simplices; title::String="", saddles::Vector=ComplexF64[])
    ax.aspect = DataAspect()
    ax.xlabel = L"\mathrm{Re}(z)"
    ax.ylabel = L"\mathrm{Im}(z)"

    xs = [real(p.coord) for p in points]
    ys = [imag(p.coord) for p in points]
    if !isempty(xs)
        xmin, xmax = minimum(xs), maximum(xs)
        ymin, ymax = minimum(ys), maximum(ys)
        dx = max(xmax - xmin, 1.0)
        dy = max(ymax - ymin, 1.0)
        range_r = range(xmin - 0.2 * dx, xmax + 0.2 * dx, length=150)
        range_i = range(ymin - 0.2 * dy, ymax + 0.2 * dy, length=150)
    else
        range_r = range(-5.0, 5.0, length=150)
        range_i = range(-5.0, 5.0, length=150)
    end

    Phi_grid = [S(Complex(tr, ti)) for tr in range_r, ti in range_i]
    contourf!(ax, range_r, range_i, -imag.(Phi_grid), colormap=:grayyellow, levels=20, extendlow=:white, extendhigh=:white)

    hlines!(ax, 0.0, color=:black, linewidth=2.5, linestyle=:dash)

    if !isempty(saddles)
        scatter!(ax, [real(s) for s in saddles], [imag(s) for s in saddles], color=:green, marker=:circle, markersize=14)
    end

    for seg in simplices
        pts = [reim(pt.coord) for pt in points[seg.coord]]
        if seg.active
            lines!(ax, pts, color=:red, linewidth=2.5)
        else
            lines!(ax, pts, color=:blue, linewidth=2.5)
        end
    end

    if !isempty(title)
        ax.title = title
    end
    return ax
end


# --------------------------------------------------------------------------
# Animation recording.
# --------------------------------------------------------------------------

# Record the dynamic deformation of a thimble under downward flow.
function record_thimble_flow_animation(S::Function, drv::Function, ξ::ComplexF64, ω::Float64, tmin::Real, tmax::Real, filename::String="thimble_flow.mp4";
    preset::Symbol=:accurate, framerate=20, keep_connected=false, saddles::Vector=ComplexF64[ξ]
)
    params = get_pl_heuristics_1d(S, drv, ξ, ω, preset)

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect(), title=L"\mathrm{Thimble\ Flow\ Evolution}", xlabel=L"\mathrm{Re}(z)", ylabel=L"\mathrm{Im}(z)")

    treals = range(Float64(tmin) * 1.5, Float64(tmax) * 1.5, length=150)
    timags = range(-abs(Float64(tmax) - Float64(tmin)) / 2, abs(Float64(tmax) - Float64(tmin)) / 2, length=150)
    Phi = [S(Complex(tr, ti)) for tr in treals, ti in timags]
    contourf!(ax, treals, timags, -imag.(Phi), colormap=:grayyellow, levels=20, extendlow=:white, extendhigh=:white)

    hlines!(ax, 0.0, color=:black, linewidth=2.5, linestyle=:dash, label=L"\mathrm{Original\ Contour}")

    if !isempty(saddles)
        scatter!(ax, [real(s) for s in saddles], [imag(s) for s in saddles], color=:green, marker=:circle, markersize=14, label=L"\mathrm{Saddles}")
    end

    active_lines = Observable(Point2f[])
    inactive_lines = Observable(Point2f[])

    linesegments!(ax, active_lines, color=:red, linewidth=2.5, label=L"\mathrm{Thimble}")
    linesegments!(ax, inactive_lines, color=:blue, linewidth=2.5, label=L"\mathrm{Flowed\ Contour}")
    axislegend(ax, position=:rt)

    (points, simplices) = initialise(Float64(tmin), Float64(tmax), params.Δinit)

    record(fig, filename, 1:params.Nflow; framerate=framerate) do i_flow
        flow_down!((S, drv), points, simplices,
            threshold=params.gradnthreshold, δ=0.015, h_threshold=params.h_threshold)

        if keep_connected
            subdivide_keep(points, simplices, params.subdividethreshold)
        else
            subdivide_rep(points, simplices, params.subdividethreshold)
        end

        act_segs = Point2f[]
        inact_segs = Point2f[]
        for sim in simplices
            p1 = points[sim.coord[1]].coord
            p2 = points[sim.coord[2]].coord
            if sim.active
                push!(act_segs, Point2f(reim(p1)))
                push!(act_segs, Point2f(reim(p2)))
            else
                push!(inact_segs, Point2f(reim(p1)))
                push!(inact_segs, Point2f(reim(p2)))
            end
        end
        active_lines[] = act_segs
        inactive_lines[] = inact_segs
    end
    return filename
end


# --------------------------------------------------------------------------
# 3D Thimble plotting.
# --------------------------------------------------------------------------

# Plot a 3D thimble mesh from 2D surface elements using a coordinate mapping.
function plot_thimble_3d(ax, simplices, coords_fn::Function;
    saddles=[],
    color=:blue,
    transparency=true,
    wireframe_color=:blue,
    markersize=14,
    markercolor=:green
)
    vertices = reduce(vcat, [
        [Point3f(coords_fn(p)) for p in tri.points]
        for tri in simplices
    ])
    faces = [TriangleFace(i, i+1, i+2) for i in 1:3:length(vertices)]
    m = normal_mesh(vertices, faces)

    mesh!(ax, m, color=color, transparency=transparency)
    wireframe!(ax, m, color=wireframe_color)

    if !isempty(saddles)
        scatter!(ax, [Point3f(coords_fn(s)) for s in saddles], color=markercolor, marker=:circle, markersize=markersize)
    end
    return ax
end
