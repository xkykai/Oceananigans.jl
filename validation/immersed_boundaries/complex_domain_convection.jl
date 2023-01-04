using Oceananigans
using GLMakie
using Printf

#####
##### Model setup
#####

Nx = 64
Ny = 1
Nz = 64

grid = RectilinearGrid(size = (Nx, Ny, Nz), 
                       x = (0, 1),
                       y = (0, 1),
                       z = (-1, 0),
                       topology = (Bounded, Periodic, Bounded))

slope(x, y) = x - 1
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))

model = NonhydrostaticModel(; grid,
                            advection = WENO(),
                            tracers=:b,
                            buoyancy=BuoyancyTracer())

# Cold blob
h = 0.05
bᵢ(x, y, z) = - exp(-(x^2 + z^2) / 2h^2)
set!(model, b=bᵢ)

#####
##### Simulation
#####

simulation = Simulation(model, Δt=1e-3, stop_iteration=10)

progress(sim) = @printf "Iter: %d, time: %s" iteration(sim) prettytime(sim) 
simulation.callbacks[:p] = Callback(progress, IterationInterval(10))

run!(simulation)

#####
##### Visualize
#####

fig = Figure(resolution=(800, 600))
ax = Axis(fig[1, 1])

b = interior(model.tracers.b, :, 1, :)
heatmap!(ax, b)

display(fig)
