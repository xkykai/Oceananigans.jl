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
closure = nothing #ScalarDiffusivity(ν=1e-3)

model = NonhydrostaticModel(; grid, closure,
                            # timestepper = :RungeKutta3,
                            advection = WENO(),
                            coriolis = FPlane(f=0.1),
                            tracers = (:b, :c),
                            buoyancy = BuoyancyTracer())

# Cold blob
h = 0.05
x₀ = 0.5
z₀ = -0.25
bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (z - z₀)^2) / 2h^2)
cᵢ(x, y, z) = ifelse(slope(x, y) < 0, 1, 0)
set!(model, b=bᵢ, c=cᵢ)

#####
##### Simulation
#####

simulation = Simulation(model, Δt=1e-2, stop_time = 10)

wall_time = Ref(time_ns())

b = model.tracers.b
B = Field(Integral(b))
compute!(B)
B₀ = B[1, 1, 1]

c = model.tracers.c
C = Field(Integral(c))
compute!(C)
C₀ = C[1, 1, 1]


function progress(sim)
    elapsed = time_ns() - wall_time[]

    compute!(B)
    Bₙ = B[1, 1, 1]
    Cₙ = C[1, 1, 1]

    @info @sprintf("Iter: %d, time: %s, wall time: %s, ΔB: %.2e, ΔC: %.2e",
                   iteration(sim), prettytime(sim), prettytime(1e-9 * elapsed),
                   (Bₙ - B₀) / B₀,
                   (Cₙ - C₀) / C₀,)

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

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B, C);
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
ct = FieldTimeSeries(filename, "c")
times = bt.times
Nt = length(times)

time_series_filename = prefix * "_time_series.jld2"
Bt = FieldTimeSeries(time_series_filename, "B")
Ct = FieldTimeSeries(time_series_filename, "C")

##
fig = Figure(resolution=(1700, 1000))

slider = Slider(fig[0, 1:4], range=1:Nt, startvalue=1)
n = slider.value

B₀ = interior(Bt[1], 1, 1, 1)
C₀ = interior(Ct[1], 1, 1, 1)

btitlestr = @lift @sprintf("Buoyancy at t = %.2f", times[$n])
wtitlestr = @lift @sprintf("Vertical velocity at t = %.2f", times[$n])
ctitlestr = @lift @sprintf("Passive tracer at t = %.2f", times[$n])
Δctitlestr = @lift @sprintf("Passive tracer anomaly at t = %.2f", times[$n])

axb = Axis(fig[1, 1], title=btitlestr)
axw = Axis(fig[1, 2], title=wtitlestr)
axc = Axis(fig[1, 3], title=ctitlestr)
axΔc = Axis(fig[1, 4], title=Δctitlestr)

axbt = Axis(fig[2, 1:2], xlabel="Time", ylabel="(∫b - ∫b₀) / ∫b₀", title="Total buoyancy anomaly")
axct = Axis(fig[2, 3:4], xlabel="Time", ylabel="(∫c - ∫c₀) / ∫c₀", title="Total passive tracer anomaly")

bn = @lift interior(bt[$n], :, 1, :)
wn = @lift interior(wt[$n], :, 1, :)
cn = @lift interior(ct[$n], :, 1, :)
Δcn = @lift interior(ct[$n], :, 1, :) .- interior(ct[1], :, 1, :)

blim = 0.2
wlim = maximum(abs, wt) / 2
clim = maximum(abs, ct) / 10
Δclim = maximum(abs, ct) / 10

heatmap!(axb, bn, colormap=:balance, colorrange=(-blim, blim))
heatmap!(axw, wn, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axc, cn, colormap=:balance, colorrange=(-clim, clim))
heatmap!(axΔc, Δcn, colormap=:balance, colorrange=(-Δclim, Δclim))

ΔB = (Bt.data[1, 1, 1, :] .- Bt.data[1, 1, 1, 1]) ./ Bt.data[1, 1, 1, 1]
t = @lift times[$n]
lines!(axbt, Bt.times, ΔB)
vlines!(axbt, t)

ΔC = (Ct.data[1, 1, 1, :] .- Ct.data[1, 1, 1, 1]) ./ Ct.data[1, 1, 1, 1]
lines!(axct, Ct.times, ΔC)
vlines!(axct, t)

display(fig)
##

record(fig, "complex_domain_convection.mp4", 1:Nt, framerate=12) do nn
    n[] = nn
end
