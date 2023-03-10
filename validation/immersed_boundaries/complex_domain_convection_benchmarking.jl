using Oceananigans
using Printf
using BenchmarkTools
using JLD2
using CairoMakie
using BenchmarkPlots
using NVTX

include("immersed_pressure_solver.jl")

function setup_grid(N)
    grid = RectilinearGrid(GPU(), Float64,
                        size = (N, N, N), 
                        halo = (5, 5, 5),
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

function setup_immersed(N)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                pressure_solver = ImmersedPoissonSolver(grid, preconditioner=true, reltol=1e-7),
                                advection = WENO(order=7),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)

    # simulation = setup_simulation(model, 2e-2 * 64 / 2 / grid.Nz, stop_iteration)
    return model
end

function setup_FFT(N)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                advection = WENO(order=7),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)

    # simulation = setup_simulation(model, 2e-2 * 64 / 2 / grid.Nz, stop_iteration)
    # time_step!(model, 0.1)
    
    return model
end

Ns = [32, 64, 128, 256]
# Ns = [32, 64, 128]

for N in Ns
    Δt = 2e-2 * 64 / 2 / N
    modelFFT = setup_FFT(N)
    modelImmersed = setup_immersed(N)
    for step in 1:3
        time_step!(modelFFT, Δt)
        time_step!(modelImmersed, Δt)
    end

    for step in 1:20
        NVTX.@range "FFT timestep N $N" begin
            time_step!(modelFFT, Δt)
        end
        NVTX.@range "Immersed timestep N $N" begin
            time_step!(modelImmersed, Δt)
        end
        @info "PCG iteration = $(modelImmersed.pressure_solver.pcg_solver.iteration)"
    end
end

# for N in Ns
#     suite["FFTBasedPoissonSolver"]["$N"] = @benchmarkable run!(simulation) setup=(simulation=setup_FFT($N, 500)) seconds=1200
#     suite["ImmersedPoissonSolver"]["$N"] = @benchmarkable run!(simulation) setup=(simulation=setup_immersed($N, 500)) seconds=1200
# end

# @info "Tuning benchmarking suite"
# @info tune!(suite)

# @info "Starting benchmarking"
# results = run(suite, verbose=true)

#=
jldopen("complex_domain_convection_benchmarking_results.jld2", "w") do file
    file["results"] = results
end

results = jldopen("complex_domain_convection_benchmarking_results.jld2", "r") do file
    file["results"]
end

results["ImmersedPoissonSolver"]["128"]

Ns = [32, 64, 128, 256]

t_median_immersed = [median(results["ImmersedPoissonSolver"]["$N"]).time / 1e9 for N in Ns]
t_median_FFT = [median(results["FFTBasedPoissonSolver"]["$N"]).time / 1e9 for N in Ns]

fig = Figure()
ax = Axis(fig[1, 1], xlabel="N", ylabel="Median time (s)", yscale=log10, xscale=log2, title="Sloped convection, GPU, 3D setup (N³ grid points)")
lines!(ax, Ns, t_median_immersed, label="Immersed solver")
lines!(ax, Ns, t_median_FFT, label="FFT solver")
axislegend(ax, position=:lt)
display(fig)
save("sloped_convection_benchmarks.png", fig, px_per_unit=4)
=#