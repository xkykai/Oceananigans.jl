using Oceananigans
using Oceananigans.Advection: div_𝐯u, div_𝐯w
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
grid = RectilinearGrid(size=(64, 64), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
for scheme in (Centered(), WENO())
    model = NonhydrostaticModel(grid; advection=scheme)
    set!(model, u = (x, z) -> 1 + cos(3x) * sin(2π * z), w = (x, z) -> sin(4π * x) * cos(π * z))
    o = Field{Face, Center, Center}(grid); parent(o) .= 1
    U = model.velocities
    Du = [div_𝐯u(i, 1, k, grid, model.advection.momentum, U, o) for i in 2:64, k in 1:64]
    Dw = [div_𝐯w(i, 1, k, grid, model.advection.momentum, U, o) for i in 1:64, k in 2:64]
    scale = maximum(abs, interior(U.u)) * 64
    println(summary(scheme), ": max|D|/scale = ", max(maximum(abs, Du), maximum(abs, Dw)) / scale)
end
