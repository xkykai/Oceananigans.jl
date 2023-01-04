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
                       halo = (4, 4, 4),
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
x₀ = 0.5
z₀ = -0.25
bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (z - z₀)^2) / 2h^2)
set!(model, b=bᵢ)

#####
##### Simulation
#####

simulation = Simulation(model, Δt=1e-2, stop_time = 1)

wall_time = Ref(time_ns())

function progress(sim)
    elapsed = time_ns() - wall_time[]
    @info @sprintf("Iter: %d, time: %s, wall time: %s",
                   iteration(sim), prettytime(sim), prettytime(1e-9 * elapsed))
    wall_time[] = time_ns()
    return nothing
end
                   
simulation.callbacks[:p] = Callback(progress, IterationInterval(100))

prefix = "complex_domain_convection"
outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                    filename = prefix * "_fields",
                                                    schedule = TimeInterval(0.1),
                                                    overwrite_existing = true)

#=
b = model.tracers.b
B = Integral(b, dims=(1, 2, 3))
simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B);
                                                          filename = prefix * "_time_series",
                                                          schedule = IterationInterval(1),
                                                          overwrite_existing = true)
=#

run!(simulation)

#####
##### Visualize
#####

filename = prefix * "_fields.jld2"
bt = FieldTimeSeries(filename, "b")
wt = FieldTimeSeries(filename, "w")
Nt = length(bt.times)

fig = Figure(resolution=(800, 600))
axb = Axis(fig[1, 1])
axw = Axis(fig[1, 2])

n = Nt
b = interior(bt[n], :, 1, :)
w = interior(wt[n], :, 1, :)
heatmap!(axb, b)
heatmap!(axw, w)

display(fig)
