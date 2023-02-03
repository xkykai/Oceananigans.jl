using Oceananigans
using GLMakie
using Printf

#####
##### Model setup
#####

Ny = 1
Nx = Nz = 64

grid = RectilinearGrid(size = (Nx, Ny, Nz), 
                       halo = (4, 4, 4),
                       x = (0, 2),
                       y = (0, 1),
                       z = (0, 1),
                       topology = (Periodic, Periodic, Bounded))

wedge(x, y) = 0.5 * min(x, 2 - x)
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(wedge))
closure = nothing #ScalarDiffusivity(ν=1e-3)

model = NonhydrostaticModel(; grid, closure,
                            # timestepper = :RungeKutta3,
                            advection = WENO(),
                            tracers = :c)

# Cold blob
cᵢ(x, y, z) = sin(π * x)
#set!(model, c=cᵢ, u=1)
set!(model, c=1, u=1)

#####
##### Simulation
#####

simulation = Simulation(model, Δt=0.1 / Nx, stop_time = 10)

wizard = TimeStepWizard(cfl=0.2, max_change=1.05)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

wall_time = Ref(time_ns())

u, v, w = model.velocities
c = model.tracers.c
C = Field(Integral(c))
compute!(C)
C₀ = C[1, 1, 1]

function progress(sim)
    elapsed = time_ns() - wall_time[]

    compute!(C)
    Cₙ = C[1, 1, 1]

    max_u = maximum(abs, u)

    @info @sprintf("Iter: %d, time: %s, wall time: %s, max(u): %.2e, ∫c: %.2e",
                   iteration(sim), prettytime(sim), prettytime(1e-9 * elapsed), max_u, Cₙ)

    wall_time[] = time_ns()

    return nothing
end
                   
simulation.callbacks[:p] = Callback(progress, IterationInterval(100))

prefix = "flow_over_wedge"

δ = ∂x(u) + ∂y(v) + ∂z(w)
p = model.pressures.pNHS
outputs = merge(model.velocities, model.tracers, (; δ, p))

simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                    filename = prefix * "_fields",
                                                    schedule = IterationInterval(1),
                                                    overwrite_existing = true)

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; C);
                                                          filename = prefix * "_time_series",
                                                          schedule = IterationInterval(1),
                                                          overwrite_existing = true)

run!(simulation)

#####
##### Visualize
#####

filename = prefix * "_fields.jld2"
ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")
ct = FieldTimeSeries(filename, "c")
δt = FieldTimeSeries(filename, "δ")
pt = FieldTimeSeries(filename, "p")
Nt = length(ct.times)

time_series_filename = prefix * "_time_series.jld2"
Ct = FieldTimeSeries(time_series_filename, "C")

fig = Figure(resolution=(1200, 600))

slider = Slider(fig[0, 1:2], range=1:Nt, startvalue=1)
n = slider.value

axu = Axis(fig[1, 1], title="Horizontal velocity")
axc = Axis(fig[1, 2], title="Tracer")
#axw = Axis(fig[1, 2], title="Vertical velocity")
axp = Axis(fig[2, 1], title="Pressure")
axδ = Axis(fig[2, 2], title="Divergence")
axt = Axis(fig[3, 1:2], title="Time series")

cn = @lift interior(ct[$n], :, 1, :)
δn = @lift interior(δt[$n], :, 1, :)
pn = @lift interior(pt[$n], :, 1, :)
un = @lift interior(ut[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)

ulim = maximum(abs, ut) / 2
plim = maximum(abs, pt[end]) / 2
dlim = 1e-14
heatmap!(axu, un, colormap=:balance, colorrange=(-ulim, ulim))
contour!(axu, pn, levels=20, color=:black, linewidth=3)
heatmap!(axc, cn, colormap=:balance, colorrange=(-0.5, 0.5))
#heatmap!(axw, wn, colormap=:balance, colorrange=(-ulim, ulim))
contourf!(axp, pn, colormap=:balance, colorrange=(-plim, plim), levels=20)
heatmap!(axδ, δn, colormap=:balance, colorrange=(-dlim, dlim))

lines!(axt, Ct.times, Ct[1, 1, 1, :])

display(fig)

# record(fig, "complex_domain_convection.mp4", 1:Nt, framerate=12) do nn
#     n[] = nn
# end
