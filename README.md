# PicardLefschetz.jl — Integration methods for highly-oscillatory integrals based on Picard–Lefschetz theory. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://anneaux.github.io/PicardLefschetz.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://anneaux.github.io/PicardLefschetz.jl/dev/)
[![Build Status](https://github.com/anneaux/PicardLefschetz.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/anneaux/PicardLefschetz.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/anneaux/PicardLefschetz.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/anneaux/PicardLefschetz.jl)


Based on Picard–Lefschetz theory, this package implements two approaches to evaluate integrals of the form
```math
\int_{R^n} ~ \mathbf{p}(\mathbf{x}) ~ \mathrm{e}^{\mathrm{i} \phi(\mathbf{x})} d\mathbf{x}
```
where $\phi(\mathbf{x})$ is analytic almost everywhere, $\mathbf{p}(\mathbf{x})$ is a slowly-varying prefactor and the integral is originally taken over the real domain $R$.
So far, the methods are implemented for one ($n=1$) and two-dimensional ($n=2$) integrals, but could be extended to higher dimensions as well.

## Numerical approaches
Please find a detailed explanation of the underlying theory as well as their implementation and a set of examples in the documentation. 
In short, the two main approaches are: 
### Approach 1: The downward flow
Flowing the integration domain into the complex domain, towards a steepest-descent manifold. This manifold is saved as a set of simplices (line segments for !D, surface elements in 2D), on which the integral can be evaluated using a quadrature of choice. 

### Approach 2: Saddle-point based methods
Finding points of the exponentiated phase function where the first derivative vanishes, i.e., $ \nabla \phi(\mathbf{x}) = 0$, and summing over relevant saddle points' contribution. 
To decide whether a saddle point is relevant we have to determine whether it's dual thimble intersects the original integration domain (the real domain).
A relevant saddles' contribution can then be calculated exactly as a quadrature along its thimble, or approximated by a Gaussian integral. The latter is known as the standard "saddle-point approximation".

## Related packages
- An efficient computation of the full integral using the downward flow (approach 1), for higher dimensions is available here: 
[PicardLefschetzIntegration.jl](https://github.com/jfeldbrugge/PicardLefschetzIntegration.jl)
- Identifying a steepest-descent integration path for 1D polynomial phase functions, is available here: [NumericalSteepestDescent.jl](https://github.com/tcaussade/NumericalSteepestDescent.jl)


## Citation
The methods in this package are partially based on [Job Feldbrugge's implementation](https://p-lpi.github.io/) and were originally developed alongside their application to integrals in attosecond science. They are explained in the respective [publication](https://journals.aps.org/pra/abstract/10.1103/d2pt-xp7x). If you use this package, please consider citing this article.
```
@article{weber2026universal,
  title = {Universal approach to saddle-point methods in attosecond science},
  author = {Weber, Anne and Feldbrugge, Job and Pisanty, Emilio},
  journal = {Phys. Rev. A},
  volume = {113},
  issue = {6},
  pages = {063111},
  numpages = {24},
  year = {2026},
  month = {Jun},
  publisher = {American Physical Society},
  doi = {10.1103/d2pt-xp7x},
  url = {https://link.aps.org/doi/10.1103/d2pt-xp7x}
}
```
Alternatively, some more in-depth explanation can be found in the author's [PhD thesis](https://arxiv.org/abs/2605.03794).


## Further developments and contribution
...are very very welcome! Feel free to open an issue or contact me directly.
