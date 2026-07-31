==A **FLowed Integration Contour (FLIC)** is the numerical object this package builds in place of an exact Lefschetz thimble: the original real domain, evolved for a finite time under the downward flow, so that it approximates $\mathcal{T}_\sigma$ closely enough to integrate directly. Building one is a short pipeline: define the domain, flow it, choose a resolution, integrate, and check that the result has converged. `PicardLefschetz.jl` exposes one function for each step.

## (a) The Integration Domain

Every FLIC starts from the original contour $\Omega$ over which

```math \int_{\Omega} P({x}),e^{iS({x})},d{x} ```

is defined, before any deformation. The package does not infer $\Omega$ from $S$; bounds must be given explicitly, either as the endpoints of a 1D interval,

```julia
tmin, tmax = -5.0, 5.0
```

or as a rectangular region in $\mathbb{C}$ for higher-dimensional problems,

```julia
domain = ComplexDomain(-5.0, 5.0, -5.0, 5.0)
```

with `make_init_points_rectangle` available if you'd rather generate the boundary vertices yourself. Either way, $\Omega$ is discretised into points and segments, giving the contour at flow time zero, before `get_thimble` touches it.

## (b) Flowing the Domain

`get_thimble` evolves that starting contour under the downward flow introduced above, tracking the deforming curve itself rather than any single endpoint, until it has converged onto (a numerical approximation of) the union of contributing thimbles. Points separate from one another as the flow stretches the contour, and the package subdivides dynamically to keep the resolution adequate as this happens.

_(animation showing this)

For the Fresnel integral $\int e^{it^2},dt$, with $S(t)=t^2$ and $S'(t)=2t$:

```julia
S(t) = t^2
drv(t) = 2t
tmin, tmax = -5.0, 5.0

points, simplices = get_thimble(S, drv, tmin, tmax; preset=:accurate)
```

`points` gives the complex vertices of the resulting contour (each flagged active or not), and `simplices` the connectivity between them:

```julia
n_points = length(points)
n_segments = count(sim -> sim.active, simplices)
println("FLIC contains $n_points points and $n_segments active segments.")
```

## (c) Choosing Flow Parameters

Rather than hand-tuning mesh spacing, subdivision thresholds, and step sizes independently, the package derives them all from a single local scale, $r_{\mathrm{osc}}$: the distance $r$ along a Hessian direction $d$ , from a saddle $\xi$ ,  at which $S$ has changed by about one oscillation,

```math |S(\xi + rd) - S(\xi)| = \frac{C}{\omega}. ```

 `get_pl_heuristics_1d` and `get_pl_heuristics_2d` compute this and the parameters it feeds:

```julia
params = get_pl_heuristics_1d(S, drv, 0.0 + 0.0im; preset=:accurate)
```

Three presets, `:fast`, `:accurate`, and `:thimble`, set sensible defaults for different trade-offs between speed, integral accuracy, and a well-converged FLIC; any individual parameter can still be overridden on top of a preset:

```julia
points, simplices = get_thimble(S, drv, tmin, tmax; params = params)
```

or a resolved `params` object can be passed through wholesale:

```julia
points, simplices = get_thimble(S, drv, tmin, tmax; params=params)
```

| Parameter            | Controls                                              |
| -------------------- | ----------------------------------------------------- |
| `Δinit`              | initial contour spacing                               |
| `subdividethreshold` | max segment length before subdivision                 |
| `flowstepfactor`     | gradient-flow step size                               |
| `gradnthreshold`     | gradient normalisation                                |
| `Nflow`              | number of flow steps                                  |
| `h_threshold`        | cutoff based on $h = \mathrm{Im}(S)$                  |
| `r_osc`              | the oscillation scale everything else is derived from |

(plot comparing an under-resolved, an over-subdivided, and a well-tuned FLIC side by side)

## (d) Integrating the Flowed Contour

Once a FLIC exists, `integrate_thimble` sums $T(z)e^{iS(z)}$ over its segments:

```julia
integral_val = integrate_thimble(S, points, simplices; prefactor = t -> 1.0)
```

or, if you'd rather not build the contour by hand first, it can flow and integrate in one call, continuing until the result stabilises:

```julia
integral_val, points, simplices = integrate_thimble(
    S, drv, tmin, tmax;
    preset=:accurate, Nmax=50, integral_accuracy=1e-7
)
```


## Keeping the Contour Connected

As the flow proceeds, pieces of the original contour can separate toward different saddles; by default the package treats these pieces independently once they split. Passing `keep_connected=true` connects the pieces and `promote_bridges=true` makes it so that the full contour is integrated as one connected object:

```julia
points, simplices = get_thimble(
    S, drv, tmin, tmax;
    preset=:accurate, keep_connected=true, promote_bridges=true
)
```

