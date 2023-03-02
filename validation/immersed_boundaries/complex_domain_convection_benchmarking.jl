using Oceananigans
using Printf
using BenchmarkTools
using JLD2

include("immersed_pressure_solver.jl")

function setup_grid(N)
    grid = RectilinearGrid(GPU(), Float64,
                        size = (N, N, N), 
                        halo = (4, 4, 4),
                        x = (0, 1),
                        y = (0, 1),
                        z = (0, 1),
                        topology = (Bounded, Periodic, Bounded))

    slope(x, y) = 1 - (x + y) / 2
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(slope))
    return grid
end

function initial_conditions!(model)
    h = 0.05
    x₀ = 0.5
    y₀ = 0.5
    z₀ = 0.75
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)
    set!(model, b=bᵢ)
end

function setup_simulation(model, Δt, stop_iteration)
    return Simulation(model, Δt=Δt, stop_iteration=stop_iteration)
end

function setup_immersed(N, stop_iteration)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                pressure_solver = ImmersedPoissonSolver(grid, preconditioner=true, reltol=1e-10),
                                advection = WENO(),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)

    simulation = setup_simulation(model, 2e-2 * 64 / 2 / grid.Nz, stop_iteration)
    return simulation
end

function setup_FFT(N, stop_iteration)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                advection = WENO(),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)

    simulation = setup_simulation(model, 2e-2 * 64 / 2 / grid.Nz, stop_iteration)
    return simulation
end

@info "Setting up benchmarking benchmarking"

suite = BenchmarkGroup()

suite["FFTBasedPoissonSolver"] = BenchmarkGroup()
suite["ImmersedPoissonSolver"] = BenchmarkGroup()

Ns = [32, 64, 128, 256]

for N in Ns
    suite["FFTBasedPoissonSolver"]["$N"] = @benchmarkable run!(simulation) setup=(simulation=setup_FFT($N, 1000)) seconds=7200
    suite["ImmersedPoissonSolver"]["$N"] = @benchmarkable run!(simulation) setup=(simulation=setup_immersed($N, 1000)) seconds=7200
end

@info "Tuning benchmarking suite"
@info tune!(suite)

@info "Starting benchmarking"
results = run(suite, verbose=true)

jldopen("complex_domain_convection_benchmarking_results.jld2", "w") do file
    file["results"] = results
end