using Oceananigans
using Oceananigans.Units
using Printf
using GLMakie

FILE_DIR = "validation/new_bickley_jet"

Lx = 4π

immersed_boundary_size = Lx
Ly = Lx + immersed_boundary_size

aspect_ratio = Ly / Lx

Nx = 16
Ny = Int(Nx * aspect_ratio)

grid = RectilinearGrid(size = (Nx, Ny, 1), 
                       halo = (4, 4, 4),
                       x = (-Lx/2, Lx/2),
                       y = (-Lx/2 - immersed_boundary_size, Lx/2),
                       z = (0, 1),
                       topology = (Periodic, Bounded, Bounded)
                       )

southern_wall(x, y) = ifelse(y < -Lx/2, 1.1, 0)

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(southern_wall))

model = NonhydrostaticModel(;
                            grid = grid,
                            buoyancy = BuoyancyTracer(),
                            tracers = (:b, :c),
                            advection = WENO()
                            )

ϵ = 0.1 # perturbation magnitude
ℓ = 0.5 # gaussian width
k = 0.5 # sinusoidal wavenumber

# Definition of the "Bickley jet": a sech(y)^2 jet with sinusoidal tracer
Ψ(y) = - tanh(y)
U(y) = sech(y)^2

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x)

uᵢ(x, y, z) = U(y) + ϵ * ũ(x, y, ℓ, k)
vᵢ(x, y, z) = ϵ * ṽ(x, y, ℓ, k)
cᵢ(x, y, z) = C(y, Ly - immersed_boundary_size)

# Note that u, v are only horizontally-divergence-free as resolution -> ∞.
set!(model, u=uᵢ, v=vᵢ, c=cᵢ, b=cᵢ)

Δt = 0.2 * 2π / Nx
stop_time = 200

simulation = Simulation(model; Δt, stop_time)

progress(sim) = @printf("Iter: %d, time: %.1f, Δt: %.1e, max|u|: %.3f, max|v|: %.3f, max|b|: %.3f, max|c|: %.3f\n",
                        iteration(sim), time(sim), sim.Δt,
                        maximum(abs, model.velocities.u),
                        maximum(abs, model.velocities.v),
                        maximum(abs, model.tracers.b),
                        maximum(abs, model.tracers.c),
                        )

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

wizard = TimeStepWizard(cfl=0.2, max_change=1.1, max_Δt=10.0)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                    filename = "$(FILE_DIR)/bickley_jet_fields_immersed",
                                                    schedule = TimeInterval(0.5),
                                                    overwrite_existing = true)

b = model.tracers.b
B = Field(Integral(b))

tracer_passive = model.tracers.c
tracer_passive_integral = Field(Integral(tracer_passive))

simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (B=B, C=tracer_passive_integral);
                                                         filename = "$(FILE_DIR)/bickley_jet_timeseries_immersed",
                                                         schedule = TimeInterval(0.5),
                                                         overwrite_existing = true)


run!(simulation)

# bt = FieldTimeSeries("$(FILE_DIR)/bickley_jet_field.jld2", "b")
# ct = FieldTimeSeries("$(FILE_DIR)/bickley_jet_field.jld2", "c")

# Bt = FieldTimeSeries("$(FILE_DIR)/bickley_jet_timeseries.jld2", "B")
# Ct = FieldTimeSeries("$(FILE_DIR)/bickley_jet_timeseries.jld2", "C")

bt = FieldTimeSeries("$(FILE_DIR)/bickley_jet_fields_immersed.jld2", "b")
ct = FieldTimeSeries("$(FILE_DIR)/bickley_jet_fields_immersed.jld2", "c")

Bt = FieldTimeSeries("$(FILE_DIR)/bickley_jet_timeseries_immersed.jld2", "B")
Ct = FieldTimeSeries("$(FILE_DIR)/bickley_jet_timeseries_immersed.jld2", "C")

Nt = length(bt.times)

##
fig = Figure(resolution=(1200, 1200))

slider = Slider(fig[0, 1:2], range=1:Nt, startvalue=1)
n = slider.value

B₀ = sum(interior(bt[1], :, :, 1)) * Lx * Ly / Nx / Ny
C₀ = sum(interior(ct[1], :, :, 1)) * Lx * Ly / Nx / Ny

Bt_sum = [sum(interior(bt[i], :, :, 1)) * Lx * Ly / Nx / Ny for i in 1:length(bt.times)]
Ct_sum = [sum(interior(ct[i], :, :, 1)) * Lx * Ly / Nx / Ny for i in 1:length(ct.times)]

ΔB_sum = Bt_sum .- B₀
ΔC_sum = Ct_sum .- C₀

# b_str = @lift string("Buoyancy, Δb = $(@sprintf("%.2e", ΔB_sum[$n]))")
# c_str = @lift string("Passive tracer, Δc = $(@sprintf("%.2e", ΔC_sum[$n]))")

# b_str = @lift string("Buoyancy, Δb (direct sum) = $(@sprintf("%.2e", (Bt_sum[$n] - B₀)))")
# c_str = @lift string("Passive tracer, Δc (direct sum) = $(@sprintf("%.2e", (Ct_sum[$n] - C₀)))")

b_str = @lift string("Buoyancy, Δb (Integral) = $(@sprintf("%.2e", (Bt[$n][1, 1, 1] - Bt[1][1, 1, 1])))")
c_str = @lift string("Passive tracer, Δc (Integral) = $(@sprintf("%.2e", (Ct[$n][1, 1, 1] - Ct[1][1, 1, 1])))")

axb = Axis(fig[1, 1], title=b_str)
axc = Axis(fig[1, 2], title=c_str)
axt = Axis(fig[2, 1:2], title="Volume-integrated time series")

bn = @lift interior(bt[$n], :, :, 1)
cn = @lift interior(ct[$n], :, :, 1)

blim = maximum(abs, bt)
clim = maximum(abs, ct)

heatmap!(axb, bn, colormap=:balance, colorrange=(-blim, blim))
heatmap!(axc, cn, colormap=:balance, colorrange=(-clim, clim))

ΔB = Bt.data[1, 1, 1, :] .- Bt.data[1, 1, 1, 1]
ΔC = Ct.data[1, 1, 1, :] .- Ct.data[1, 1, 1, 1]

lines!(axt, Bt.times, ΔB, label="ΔB, Field(Integral(b))")
# lines!(axt, Bt.times, ΔC, label="ΔC, Field(Integral(c))")

# lines!(axt, bt.times, ΔB_sum, label="ΔB, direct sum")
# lines!(axt, bt.times, ΔC_sum, label="ΔC, direct sum")

# lines!(axt, Bt.times, Bt.data[1, 1, 1, :], label="Buoyancy tracer, Field(Integral(b))")
# lines!(axt, Bt.times, Ct.data[1, 1, 1, :], label="Passive tracer, Field(Integral(c))")

# lines!(axt, bt.times, Bt_sum, label="Buoyancy tracer, direct sum")
# lines!(axt, bt.times, Ct_sum, label="Passive tracer, direct sum")

axislegend()


display(fig)
##
record(fig, "$(FILE_DIR)/bickley_jet_immersed.mp4", 1:Nt, framerate=30) do nn
    n[] = nn
end

##