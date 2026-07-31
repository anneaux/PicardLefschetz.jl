```@meta
CurrentModule = PicardLefschetz
```
# PicardLefschetz.jl

PicardLefschetz.jl is a Julia package which implements 1D and 2D methods to numerically apply Picard Lefschetz theory to highly oscillatory integrals. This 
package can be used to calculate the integrals themselves, or to calculate the thimbles, the dual thimbles, the saddle points, and their intersection numbers. 

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