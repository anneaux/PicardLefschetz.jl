The `PicardLefschetz.jl` package implements several methods that allow the user to directly with saddle points instead of the total FLIC.

---
# S.1 Finding Saddle Points
The package implements two main methods to find saddle points. One can be used to find a single saddle point, using Newton Raphson iterations over a single initial guess, and the other using Newton Raphson iterations over a real domain to find all saddle points in that domain.
- `solve_first_derivative`: This function can be used to find a single saddle, given an initial guess. It uses Newton-Raphson iteration to find the saddle point, and once it converges to the arbitrary accuracy provided the user, returns the saddle point as a `Saddle`.
- `find_saddles`: This function can be used to find all saddle points in a real domain. It initialises a Sobol sequence over the provided real domain, and then uses Newton-Raphson iteration to evolve the points about the saddle points, until they converge to a saddle point to an arbitrary accuracy. The saddle points are then deduplicated from the list, and returned as a `Vector{Saddle}`. 
## Examples
```julia
# Insert example for solve_first_derivative here

# Insert example for find_saddles here
```

---
# S.2 Calculating the Thimbles and Dual Thimbles
Before we check whether a saddle point contributes and calculate its contribution, we must first discuss how to calculate the thimble and dual thimble manifolds. 
## S.2.1 Calculating the Thimble Manifold
This package implements two different approaches to calculating the thimble for an integration contour, one being the continuous deformation of the FLIC (flowed integration contour), $\gamma_\lambda(x_0)$, into the thimble using gradient descent, and the other being the individual calculation of the thimbles using gradient descent at a saddle point. The saddle point approach is to first calculate the saddle point, and then initialise a contour on the Lefschetz thimble. This contour is then evolved along the thimble until an arbitrary cutoff height. The pullback of the integration contour over the thimble, which is the contour we initialise and evolve over the thimble, satisfies the following differential equation
```math
\frac{d\gamma^\sigma_\lambda(x_0)}{d\lambda} = -\nabla h(\gamma^\sigma_\lambda(x_0))
```
where $\gamma^\sigma_\lambda$ is the $\sigma$-th segment of the deformed integration contour, associated with the $\sigma$-th saddle point. The thimble is the contour obtained in the limit $\gamma(x_0) = \lim_{\lambda\to\infty}\gamma_\lambda(x_0)$. Functionality to compute the integral over the deformed integration contour is defined in section S.4.
- `get_thimble!`: This method calculates the individual thimble associated with a saddle point, using gradient descent on a thimble contour that is initialised about the saddle point. It evolves the contour along the thimble up to an arbitrary cutoff height, or for an arbitrary number of flow steps. In 1D, this computes a line, and in 2D, this computes a mesh, both of which are represented using a point cloud, a list of simplices which map the indices of the points in the point cloud to each other, forming a simplex mesh.
- `get_thimbles`: This function calculates the saddle points in a given real domain, using the `find_saddles` function, and calculates the thimbles for each of the found saddle points using the `get_thimble!` method. 
## S.2.2 Calculating the Thimble Boundary Manifold
The thimble boundary is computed in the same manner as the thimble, however, instead of providing the result of the pullback in the limit of $\lambda \to \infty$, we provide just the pullback itself, evaluated at the value of $\lambda$ at the height cutoff provided by the user. The thimble boundary is useful particularly when quantifying the numerical error that is obtained by performing numerical integration over the deformed integration contour. 
- `get_thimble_boundary!`: This method uses the same algorithm as `get_thimble!`, however does not integrate the pullback over the thimble. In 1D, it returns two points which are the boundary of the 1D thimble manifold, and in 2D, it returns a 1D line along the thimble boundary, which is the boundary of the 2D thimble manifold. They are returned as a point cloud, which in the 2D case is represented as a point cloud and a vector of indexing simplices, in the same manner as `get_thimble!`.
- `get_thimble_boundaries`: This function finds all the saddle points in a given real domain using `find_saddles`, and iteratively computes their thimble boundaries using the `get_thimble_boundary!` method.
## Examples
```julia
# Insert example for get_thimble! here

# Insert example for get_thimbles here

# Insert example for get_thimble_boundary! here

# Insert example for get_thimble_boundaries here
```
## S.2.3 Calculating the Dual Thimble Manifold
`PicardLefschetz.jl` also implements the ability to calculate the dual thimble for given saddle point, using the same techniques as the thimble calculation per saddle point. The core algorithm is gradient ascent, which is simply the gradient descent algorithm with just the sign changed. The equation that is used for the gradient ascent algorithm takes the following form
```math
\frac{d\gamma^\sigma_\lambda(x_0)}{d\lambda} = \nabla h(\gamma^\sigma_\lambda(x_0))
```
which evolves a contour initialised on the dual Lefschetz thimble along the dual thimble until an arbitrary height cutoff. The same convention is used as in section S.2.1. The dual thimble is used to determine whether the saddle point contributes, as well as the intersection number of the saddle point's thimble. The intersection number is a topological invariant that is required to compute the integration contour, deformed over the Lefschetz thimbles. 
- `get_dual_thimble!`: This method uses gradient ascent to calculate the dual thimble manifold, but evolving a dual thimble contour initialised about the saddle point. This dual thimble contour is evolved until an arbitrary height cutoff is reached. The resulting dual thimble mesh, which in 1D is a line, and in 2D is a surface, is represented as a point cloud and a vector of indexing simplices. 
- `get_dual_thimbles`: This function calculates the saddle points using `find_saddles`, and then iteratively calculates the dual thimbles associated with each saddle point using the `get_dual_thimble!` method. 
## S.2.4 Calculating the Dual Thimble Boundary Manifold
The dual thimble boundary is calculated using the same prescription as the dual thimble itself, and transitively, the thimble. The push forward is calculated without taking the limit  $\lambda \to \infty$, which is then evaluated at the value of $\lambda$ at the height cutoff determined by the user.
- `get_dual_thimble_boundary!`: This method uses the same algorithm as the `get_dual_thimble!` method, returning the dual thimble boundary manifold. In 1D, it returns two points, which are the dual thimble boundary of the 1D dual thimble curve, and in 2D, it returns the 1D curve of the dual thimble boundary to the 2D dual thimble surface. These are represented using a point cloud and a vector of indexing simplices.
- `get_dual_thimble_boundaries`: This function finds the saddles points in a given real domain using `find_saddles`, and then iteratively computes their dual thimble boundaries using `get_dual_thimble_boundary!`. 
## Examples
```julia
# Insert example for get_dual_thimble! here

# Insert example for get_dual_thimbles here

# Insert example for get_dual_thimble_boundary! here

# Insert example for get_dual_thimble_boundaries here
```
---
# S.3 Checking Saddle Contribution
The package also implements methods to check the contribution for a given saddle point. In Picard–Lefschetz theory, a saddle point only contributes if the dual thimble (also referred to as the unstable thimble in the literature) intersects the original integration domain that original integral contour lies in. If the dual thimble intersects the original integration domain, then the intersection number of saddle point's thimble is non-zero. This follows from the theorem that states that the integration contour deformation into the Lefschetz thimbles takes the form
```math
\mathbb{R}^n \cong \sum_\sigma n_\sigma \mathcal{T}_\sigma
```
where $n_\sigma$ is the intersection number. This package implements a simple check that just checks whether the saddle point contributes by calculating whether there exists an intersection point of the dual thimble with the original integration domain. It also contains the algorithm required to compute the intersection number itself. 
- `check_contribution!`: This method runs the gradient ascent solver for evolving a dual thimble contour along the dual thimble until the contour crosses the original integration domain, or reaches a cutoff height. The dual thimble, in the case that the saddle point contributes, has one direction where the imaginary component globally decreases, and another where the imaginary component globally increases. Therefore, cutoff height applies to the globally increasing direction, and the globally decreasing direction is followed until the intersection is detected, for which the intersection point is also calculated. 
## S.3.1 Intersection Number Calculation
`PicardLefschetz.jl` contains the algorithm required to compute the intersection number of a given saddle point's thimble. 
The intersection number can take one of three values $n_\sigma \in \{-1, 0, 1\}$. We first calculate the tangent vectors to the dual thimble at the saddle point. 
Then, we push this tangent vector forward (using the gradient-ascent flow) towards the intersection point of dual thimble with the original integration domain (OID).
The tangent space to the OID is isomorphic to the OID itself (provided that the OID is Euclidean, which this package assumes).
Thus, the tangent vectors to the basis of the OID are the basis *of* the OID, i.e. in 2D, the basis embedded in the ambient complex space takes the form
```math
\begin{aligned}
e_1 &= \begin{pmatrix}1 \\ 0 \\ 0 \\ 0\end{pmatrix} &e_2 = \begin{pmatrix}0 \\ 0 \\ 1 \\ 0\end{pmatrix} \\
\end{aligned}
```
and in 1D, the basis embedded in the ambient complex space takes the form
```math
\begin{aligned}
e &= \begin{pmatrix}1 \\ 0\end{pmatrix}
\end{aligned}
```
Using the tangent vectors of the dual thimble pushed forward to the intersection point, we form the following matrix:
```math
M = \begin{pmatrix}e_1 & e_2 & u_1 & u_2\end{pmatrix}
```
and the intersection number becomes the sign of the determinant of this matrix:
```math
n_\sigma = \text{sgn}(\det{M})
```
- `get_intersection_number!`: This method actually computes the intersection number using the algorithm detailed in section S.3.1. By directly computing the intersection number, we can determine whether the thimble contributes by checking whether or not the intersection number is 0. If the intersection number for a given saddle point is 0, then it does not contribute to the integral. Otherwise, we include it in the computation of the integral.
## Examples
```julia
# Insert example for check_contribution! here

# Insert example for get_intersection_number! here
```

---
# S.4 Integrating using Saddle-Point Approaches
The main goal of Picard–Lefschetz theory is to calculate the highly oscillatory integrals of the form:
```math
\int P(x)e^{iS(x)}\,dx
```
For this, the package implements techniques that rely on the continuous deformation of the integration contour, or by using the Saddle Point Method approximation of the integral, where we assume that the contribution to integral by the saddle point is Gaussian, which is well suited to problems which involve the saddle points being very spread out. The approximation follows from the deformation that's prescribed by Picard–Lefschetz theory
```math
\begin{aligned}
\mathbb{R}^n \cong \sum_\sigma n_\sigma \mathcal{T}_\sigma \implies \int_{\mathbb{R}^n}P(x)e^{iS(x)}\,dx &= \sum_\sigma n_\sigma\int_{\mathcal{T}_\sigma}P(z)e^{iS(z)}\,dz \\
&\approx \sum_\sigma n_\sigma P(z_\sigma)\sqrt{\frac{(2\pi i)^N}{\det{S^{\prime\prime}(z_\sigma)}}}e^{iS(z_\sigma)}
\end{aligned}
```
where $S^{\prime\prime}(z_\sigma)$ is the Hessian of the action evaluated at the saddle point. This approximation follows by Taylor expanding the integrand about the saddle point, and taking the approximation of the second order. The SPM approximation also includes a separate technique to calculate the intersection number by comparing the integral calculated using the SPM method to a Gaussian integral and calculating the sign. This technique, along with the SPM approximation, is suited for problems with no saddle point clustering, i.e. no saddle points approaching Stokes' transitions. 