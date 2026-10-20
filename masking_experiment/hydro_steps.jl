using Oceananigans
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: Vᶜᶜᶜ
using Serialization
using Printf

# Stratified flow over a stepped sill whose geometry is identical at every resolution:
# risers on x grid lines, step tops on (GridFittedBottom) or between (PartialCellBottom) z grid lines
outdir, bottom_name, N = ARGS[1], ARGS[2], parse(Int, ARGS[3])
mkpath(outdir)

U, N² = 0.1, 1.0
tops = bottom_name == "full" ? (0.125, 0.25, 0.375) : (0.13, 0.26, 0.39141)

function sill(x)
    h₁, h₂, h₃ = tops
    return 0.25 ≤ x < 0.3125 ? h₁ :
           0.3125 ≤ x < 0.375 ? h₂ :
           0.375 ≤ x < 0.625 ? h₃ :
           0.625 ≤ x < 0.6875 ? h₂ :
           0.6875 ≤ x < 0.75 ? h₁ : 0.0
end

Bottom = bottom_name == "full" ? GridFittedBottom : PartialCellBottom
grid = RectilinearGrid(size=(N, N), x=(0, 1), z=(0, 1), halo=(7, 7), topology=(Periodic, Flat, Bounded))
grid = ImmersedBoundaryGrid(grid, Bottom(sill))

model = HydrostaticFreeSurfaceModel(grid; momentum_advection=WENOVectorInvariant(), tracer_advection=WENO(),
                                    tracers=:b, buoyancy=BuoyancyTracer(), free_surface=ImplicitFreeSurface(),
                                    closure=ScalarDiffusivity(ν=1e-4, κ=1e-4))
set!(model, u=U, b=(x, z) -> N² * z)

simulation = Simulation(model; Δt=0.05 / N, stop_time=4, verbose=false)
conjure_time_step_wizard!(simulation, cfl=0.2)

u, v, w = model.velocities
b = model.tracers.b
snapshots = Dict{Float64, Any}()
snapshot(sim) = snapshots[round(time(sim), digits=6)] = (; u = Array(interior(u, :, 1, :)), w = Array(interior(w, :, 1, :)),
                                                           b = Array(interior(b, :, 1, :)))
add_callback!(simulation, snapshot, SpecifiedTimes(2, 4))
elapsed = @elapsed run!(simulation)

volume = [inactive_cell(i, 1, k, grid) ? 0.0 : Vᶜᶜᶜ(i, 1, k, grid) for i in 1:N, k in 1:N]
z = collect(znodes(grid, Center()))
x = collect(xnodes(grid, Center()))
fluid_heights = [sum(volume[i, :]) * N for i in 1:N]
geometry_error = maximum(abs, fluid_heights .- (1 .- sill.(x)))

serialize(joinpath(outdir, "steps_$(bottom_name)_$N.jls"), (; snapshots, volume, z, N))
@printf("%s N=%d done in %.0f s, max|u - U| = %.3f, geometry error = %.1e\n",
        bottom_name, N, elapsed, maximum(abs, snapshots[4.0].u .- U), geometry_error)
