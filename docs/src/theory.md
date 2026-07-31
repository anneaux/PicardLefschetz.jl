Highly oscillatory integrals of the form

```math
\int_{U\subset\mathbb{R}^n} P(x)\,e^{iS(x)}\,dx
```

arise throughout physics and applied mathematics when a slowly varying amplitude $T(x)$ is modulated by a rapidly oscillating phase $S(x)$, with applications ranging from path integrals in quantum field theory to wave scattering, population dynamics, and option pricing. Direct quadrature becomes numerically unstable due to severe cancellation from these oscillations, and when the phase $S(x)$ is real-valued, as is common in physical applications, the resulting integral is often only conditionally convergent.

*(plot showing highly oscillating integrand)*

To overcome this, Picard–Lefschetz theory analytically continues the phase into complex space and deforms the original integration contour onto a set of geometrically distinguished contours called **Lefschetz thimbles**. Write

```math
iS(z)=h(z)+iH(z),
```

so

```math
e^{iS}=e^{h}e^{iH},
```

where $e^{iH}$ carries the oscillation and $e^{h}$ controls the magnitude. We seek contours along which $H$ is constant, so the oscillation disappears, while $h$ decreases, so the integrand decays and the integral becomes absolutely convergent.

*(plot showing same integrand is now smooth)*

The Cauchy–Riemann equations give

```math
\nabla h\cdot\nabla H=0,
```

so the gradient flow of $h$ is tangent to the level sets of $H$, i.e., steepest-descent trajectories preserve the phase while maximising decay. The flow has stationary points precisely where $S'(z)=0$, namely the saddle points $z_\sigma$. These organise the geometry: every Lefschetz thimble $\mathcal{T}_\sigma$ is attached to a saddle. Picard–Lefschetz theory decomposes the original contour into a sum of these thimbles, weighted by the intersection numbers $n_\sigma\in\mathbb{Z}$,

```math
I=\sum_\sigma n_\sigma\int_{\mathcal{T}_\sigma}P(z)\,e^{iS(z)}\,dz.
```

*(plot showing ascent and descent thimbles going through some saddles on a contour plot, label each thimble with its intersection number and whatever else)*

This package constructs the $\mathcal{T}_\sigma$ in two complementary ways. The **downward-flow method** flows the *entire* original contour along

```math
\frac{dz}{d\lambda}
=
-\left(\frac{\partial (iS)}{\partial z}\right)^*,
```

converging as $\lambda\to\infty$ onto the union of contributing thimbles. Unlike standard gradient descent, the object of interest is not the endpoint of the flow, but the continuously deformed contour traced by the collection of points as they are transported under the flow.

The **saddle-point method** instead finds each $z_\sigma$ first, then constructs its thimble directly — either exactly, by following steepest descent outward from $z_\sigma$, or approximately via the Gaussian formula

```math
\int_{\mathcal{T}_\sigma} P(z)e^{iS(z)}\,dz
\approx
P(z_\sigma)\sqrt{\frac{(i2\pi)^N}{\det S''(z_\sigma)}}\,e^{iS(z_\sigma)}.
```

We then determine the intersection number $n_\sigma$: first checking whether the steepest-ascent path meets the original contour, and if so, where; the sign of $n_\sigma$ is then found either by comparing the sign of the numerically integrated contribution against the Gaussian approximation above, or from how the flow maps nearby tangent vectors at the intersection point.

Once $\mathcal{T}_\sigma$ and $n_\sigma$ are known, integrating $P(z)e^{iS(z)}$ over each thimble and weighting by its intersection number gives that saddle's contribution to $I$.

Downward flow is practical to carry out directly in 1D but becomes numerically intractable in higher dimensions, where the saddle-point construction is the practical route to building $\mathcal{T}_\sigma$.

Picard–Lefschetz theory extends the contour deformation ideas of Cauchy's theorem: deformation through holomorphic regions leaves $I$ unchanged, making the thimble decomposition exact regardless of how $\mathcal{T}_\sigma$ is obtained. The Gaussian formula deteriorates as saddles coalesce ($\det S''\to0$), so the numerical constructions implemented in this package compute the thimble geometry directly, capturing behaviour beyond the local saddle-point approximation.