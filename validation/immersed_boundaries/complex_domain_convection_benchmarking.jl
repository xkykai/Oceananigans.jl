using Oceananigans
using Printf
using JLD2
using NVTX
using Statistics

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
    z₀ = 0.55
    bᵢ(x, y, z) = - exp(-((x - x₀)^2 + (y - y₀)^2 + (z - z₀)^2) / 2h^2)
    set!(model, b=bᵢ)
end

function setup_simulation(model, Δt, stop_iteration)
    return Simulation(model, Δt=Δt, stop_iteration=stop_iteration)
end

function setup_immersed_FFTprec(N)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                pressure_solver = ImmersedPoissonSolver(grid, preconditioner="FFT", reltol=1e-7),
                                advection = WENO(order=7),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

function setup_immersed_noprec(N)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                pressure_solver = ImmersedPoissonSolver(grid, preconditioner=nothing, reltol=1e-7),
                                advection = WENO(order=7),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)
    return model
end

function setup_immersed_MITgcmprec(N)
    grid = setup_grid(N)

    model = NonhydrostaticModel(; grid,
                                pressure_solver = ImmersedPoissonSolver(grid, preconditioner=MITgcmPreconditioner(), reltol=1e-7),
                                advection = WENO(order=7),
                                coriolis = FPlane(f=0.1),
                                tracers = (:b),
                                buoyancy = BuoyancyTracer())

    initial_conditions!(model)
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
    return model
end

Ns = [32, 64, 128, 160, 192, 224, 256]

Δt = 2e-2 * 64 / 2 / maximum(Ns)
nsteps = 500

@info "Benchmarking FFT solver"
for N in Ns
    model = setup_FFT(N)

    for step in 1:3
        time_step!(model, Δt)
    end

    for step in 1:nsteps
        NVTX.@range "FFT timestep, N $N" begin
            time_step!(model, Δt)
        end
    end
end

@info "Benchmarking FFT preconditioner"
PCG_N_FFTprec = zeros(length(Ns))

for (i, N) in pairs(Ns)
    model = setup_immersed_FFTprec(N)

    for step in 1:3
        time_step!(model, Δt)
    end

    PCG_iters_FFTprec = zeros(nsteps)

    for step in 1:nsteps
        NVTX.@range "Immersed timestep, FFT preconditioner N $N" begin
            time_step!(model, Δt)
        end
        @info "PCG iteration (FFT preconditioner) = $(model.pressure_solver.pcg_solver.iteration)"
        PCG_iters_FFTprec[step] = model.pressure_solver.pcg_solver.iteration
    end
    @info "Mean PCG iteration (FFT preconditioner) = $(mean(PCG_iters_FFTprec))"
    PCG_N_FFTprec[i] = mean(PCG_iters_FFTprec)
end

@info "Benchmarking no preconditioner"
PCG_N_noprec = zeros(length(Ns))

for (i, N) in pairs(Ns)
    model = setup_immersed_noprec(N)

    for step in 1:3
        time_step!(model, Δt)
    end
    PCG_iters_noprec = zeros(nsteps)

    for step in 1:nsteps
        NVTX.@range "Immersed timestep, no preconditioner N $N" begin
            time_step!(model, Δt)
        end

        @info "PCG iteration (no preconditioner) = $(model.pressure_solver.pcg_solver.iteration)"
        PCG_iters_noprec[step] = model.pressure_solver.pcg_solver.iteration
    end
    @info "Mean PCG iteration (no preconditioner) = $(mean(PCG_iters_noprec))"
    PCG_N_noprec[i] = mean(PCG_iters_noprec)
end

@info "Benchmarking MITgcm preconditioner"
PCG_N_MITgcmprec = zeros(length(Ns))

for (i, N) in pairs(Ns)
    model = setup_immersed_MITgcmprec(N)

    for step in 1:3
        time_step!(model, Δt)
    end
    PCG_iters_MITgcmprec = zeros(nsteps)

    for step in 1:nsteps
        NVTX.@range "Immersed timestep, MITgcm preconditioner N $N" begin
            time_step!(model, Δt)
        end

        @info "PCG iteration (MITgcm preconditioner) = $(model.pressure_solver.pcg_solver.iteration)"
        PCG_iters_MITgcmprec[step] = model.pressure_solver.pcg_solver.iteration
    end
    @info "Mean PCG iteration (MITgcm preconditioner) = $(mean(PCG_iters_MITgcmprec))"
    PCG_N_MITgcmprec[i] = mean(PCG_iters_MITgcmprec)
end

jldsave("PCG_N_iters.jld2"; FFTprec=PCG_N_FFTprec, noprec=PCG_N_noprec, MITgcmprec=PCG_N_MITgcmprec)


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