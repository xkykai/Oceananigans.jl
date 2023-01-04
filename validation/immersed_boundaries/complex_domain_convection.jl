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
                            coriolis = FPlane(f=0.1),
                            tracers = :b,
                            buoyancy = BuoyancyTracer())

# Cold blob
h = 0.05
x₀ = 0.5
z₀ = -0.25
bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (z - z₀)^2) / 2h^2)
set!(model, b=bᵢ)

#####
##### Simulation
#####

simulation = Simulation(model, Δt=1e-2, stop_time = 10)

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

b = model.tracers.b
B = Field(Integral(b))
simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B);
                                                          filename = prefix * "_time_series",
                                                          schedule = IterationInterval(1),
                                                          overwrite_existing = true)

run!(simulation)

#####
##### Visualize
#####

filename = prefix * "_fields.jld2"
bt = FieldTimeSeries(filename, "b")
wt = FieldTimeSeries(filename, "w")
Nt = length(bt.times)

fig = Figure(resolution=(1200, 600))

slider = Slider(fig[2, 1:2], range=1:Nt, startvalue=1)
n = slider.value

B₀ = sum(interior(bt[1], :, 1, :)) / (Nx * Nz)
titlestr = @lift string("Buoyancy, Δb = ",
                        @sprintf("%.2e %%", (sum(interior(bt[$n], :, 1, :)) / (Nx * Nz) - B₀) / B₀))

axb = Axis(fig[1, 1], title=titlestr)
axw = Axis(fig[1, 2], title="Vertical velocity")

bn = @lift interior(bt[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)

wlim = maximum(abs, wt) / 2
heatmap!(axb, bn, colormap=:balance, colorrange=(-0.5, 0.5))
heatmap!(axw, wn, colormap=:balance, colorrange=(-wlim, wlim))

display(fig)

record(fig, "complex_domain_convection.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
