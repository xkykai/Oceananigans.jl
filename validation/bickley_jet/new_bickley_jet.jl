using Oceananigans
using Oceananigans.Units
using Printf
using GLMakie

include("bickley_utils.jl")

function progress(sim)
    model = sim.model
    grid = model.grid
    c = model.tracers.c
    ∫c_int = compute!(Field(Integral(c)))[1, 1, 1]

    msg1 = @sprintf("Iter: %d, time: %.1f, Δt: %.1e, max|u|: %.3f, max|v|: %.3f, max|c|: %.3f",
                    iteration(sim), time(sim), sim.Δt,
                    maximum(abs, model.velocities.u),
                    maximum(abs, model.velocities.v),
                    maximum(abs, model.tracers.c))

    msg2 = @sprintf(", ∫c int: %.2e", ∫c_int)

    @info msg1 * msg2

    return nothing
end

function bickley_jet_simulation(grid, prefix="bickley_jet")

    model = NonhydrostaticModel(; grid,
                                timestepper = :RungeKutta3,
                                tracers = :c,
                                advection = WENO())

    set_bickley_jet!(model)

    Lx = grid.Lx
    Nx = grid.Nx
    Δt = 0.2 * Lx / Nx
    stop_time = 200

    simulation = Simulation(model; Δt, stop_time)

    
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    wizard = TimeStepWizard(cfl=0.8, max_change=1.1, max_Δt=10.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    u, v, w = model.velocities
    ζ = ∂x(v) - ∂y(u)
    δ = ∂x(u) + ∂y(v) + ∂z(w)
    outputs = merge(model.velocities, model.tracers, (; ζ, δ))

    simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                        filename = "$(prefix)_fields",
                                                        array_type = Array{Float64},
                                                        schedule = TimeInterval(0.5),
                                                        overwrite_existing = true)

    c = model.tracers.c
    C = Field(Integral(c))

    simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; C),
                                                              filename = "$(prefix)_timeseries",
                                                              schedule = TimeInterval(0.5),
                                                              overwrite_existing = true)

    return simulation
end

#####
##### Non-immersed simulation
#####

Nx = 64
Ny = 64

Lx = 4π
Ly = 4π

halo = (4, 4, 4)
x = (-Lx/2, Lx/2)
y = (-Ly/2, Ly/2)
z = (0, 1)
topology = (Periodic, Bounded, Bounded)

grid = RectilinearGrid(; size = (Nx, Ny, 1), halo, x, y, z, topology)

#simulation = bickley_jet_simulation(grid, "bickley_jet")
#run!(simulation)

#####
##### Immersed simulation
#####

immersed_boundary_width = Lx/2
immersed_Ly = Lx + immersed_boundary_width
immersed_aspect_ratio = immersed_Ly / Lx
immersed_Ny = Int(Nx * immersed_aspect_ratio)
immersed_y = -Lx/2 - immersed_boundary_width, Lx/2

underlying_grid = RectilinearGrid(; halo, x, z, topology,
                                  size = (Nx, immersed_Ny, 1),
                                  y = immersed_y)

southern_wall(x, y) = ifelse(y < -Lx/2, 1.1, 0)
immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(southern_wall))

#immersed_simulation = bickley_jet_simulation(immersed_grid, "immersed_bickley_jet")
#run!(immersed_simulation)

ct = FieldTimeSeries("immersed_bickley_jet_fields.jld2", "c")
vt = FieldTimeSeries("immersed_bickley_jet_fields.jld2", "v")
ζt = FieldTimeSeries("immersed_bickley_jet_fields.jld2", "ζ")
δt = FieldTimeSeries("immersed_bickley_jet_fields.jld2", "δ")
Ct = FieldTimeSeries("immersed_bickley_jet_timeseries.jld2", "C")

ζt_not_immersed = FieldTimeSeries("bickley_jet_fields.jld2", "ζ")
Ct_not_immersed = FieldTimeSeries("bickley_jet_timeseries.jld2", "C")

times = ct.times
Nt = length(times)

#####
##### Visualize results
#####

fig = Figure(resolution=(1200, 1200))


axc      = Axis(fig[1, 1], title="Passive tracer concentration")
axz1     = Axis(fig[1, 2], title="Immersed vorticity")
axz2     = Axis(fig[1, 3], title="Not immersed vorticity")
axe      = Axis(fig[1, 4], title="Vorticity error")
axd      = Axis(fig[1, 5], title="Divergence")
axt      = Axis(fig[2, 1:5], title="Volume-integrated passive tracer")
slider = Slider(fig[0, 1:5], range=1:Nt, startvalue=1)
n = slider.value

ζjj = (immersed_Ny-Ny+1):immersed_Ny+1
cjj = (immersed_Ny-Ny+1):immersed_Ny

cn = @lift interior(ct[$n], :, cjj, 1)
clim = maximum(abs, ct)
heatmap!(axc, cn, colormap=:balance, colorrange=(-clim, clim))

ζn = @lift interior(ζt[$n], :, ζjj, 1)
ζlim = maximum(abs, ζt)
heatmap!(axz1, ζn, colormap=:balance, colorrange=(-ζlim, ζlim))

ζn_not_immersed = @lift interior(ζt_not_immersed[$n], :, :, 1)
ζlim = maximum(abs, ζt_not_immersed)
heatmap!(axz2, ζn_not_immersed, colormap=:balance, colorrange=(-ζlim, ζlim))

En = @lift interior(ζt[$n], :, ζjj, 1) .- interior(ζt_not_immersed[$n], :, :, 1)
Elim = maximum(abs, ζt) / 10
heatmap!(axe, En, colormap=:balance, colorrange=(-Elim, Elim))

δn = @lift interior(δt[$n], :, :, 1)
δlim = 1e-15
heatmap!(axd, δn, colormap=:balance, colorrange=(-δlim, δlim))

lines!(axt, times, interior(Ct, 1, 1, 1, :))
lines!(axt, times, interior(Ct_not_immersed, 1, 1, 1, :))

display(fig)

#record(fig, "bickley_jet_direct_sum.mp4", 1:Nt, framerate=30) do nn
#    n[] = nn
#end
