```@meta
CurrentModule = PicardLefschetz
```
# PicardLefschetz.jl

PicardLefschetz.jl is a Julia package which implements the ideas of Picard–Lefschetz theory as numerical approaches to highly-oscillatory integrals.
For integrals in 1D and 2D, this package can be used to calculate the integrals themselves, or to find the Lefschetz thimbles, the dual thimbles, the saddle points, as well as their intersection numbers. 

## Installation
This package can be installed by running the following command in the Julia REPL:
```
julia> ]
pkg> add PicardLefschetz
```
or
```julia
using Pkg
Pkg.add("PicardLefschetz")
```