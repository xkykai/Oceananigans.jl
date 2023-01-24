using Printf
using Statistics
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom   

include("../bickley_jet/bickley_utils.jl")

function run_bickley_jet(grid, experiment_name;
                         output_time_interval = 2,
                         stop_time = 200,
                         advection = WENO())

    model = NonhydrostaticModel(; grid, advection
                                timestepper = :RungeKutta3,
                                tracers = :c)

    # ** Initial conditions **
    #
    # u, v: Large-scale jet + vortical perturbations
    #    c: Sinusoid
    
    set_bickley_jet!(model)

    wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=10.0)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

    simulation = Simulation(model, Δt=1e-2, stop_time=stop_time)

    progress(sim) = @info @sprintf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f",
                                   iteration(sim), prettytime(sim), prettytime(sim.Δt),
                                   maximum(abs, model.velocities.u))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

    # Output: primitive fields + computations
    u, v, w = model.velocities
    ζ = ∂x(v) - ∂y(u)

    outputs = merge(model.velocities, model.tracers, (; ζ=ζ))

    simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                          schedule = TimeInterval(output_time_interval),
                                                          filename = experiment_name,
                                                          overwrite_existing = true)

    run!(simulation)

    return nothing
end

Nh = 32
halo = (4, 4, 4)
x = (0, 4π)
z = (0, 1)
topology = (Periodic, Bounded, Bounded)

# Two grids 
grid = RectilinearGrid(arch; halo, x, z, topology,
                       size=(Nh, Nh, 1),
                       y = (-2π, 2π))

underlying_grid = RectilinearGrid(arch; halo, x, z, topology,
                                  size = (Nh, 3Nh/2, 1), 
                                  y = (-4π, 2π))

wall(x, y) = ifelse(y > -2π, 0, 1)
immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(wall))

for (grid, name) in zip([ordinary_grid, immersed_grid], ["ordinary", "immersed"])
    experiment_name = "bickley_jet_" * name
    run_bickley_jet(grid, experiment_name)
end

