using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Serialization
using Printf

outdir = ARGS[1]
mkpath(outdir)

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20

# Inviscid, unforced flow in a closed box: kinetic energy must be conserved.
function kinetic_energy_case(immersed_boundary; N=64, stop_time=4, cfl=0.2)
    grid = RectilinearGrid(size=(N, N), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol=1e-12, abstol=0, maxiter=1000)
    model = NonhydrostaticModel(ibg; advection=Centered(), pressure_solver)

    σ = 0.08
    vortex(x, z, x₀, z₀) = exp(-((x - x₀)^2 + (z - z₀)^2) / 2σ^2)
    uᵢ(x, z) = - ((z - 0.7) / σ^2) * (- vortex(x, z, 0.35, 0.7) + vortex(x, z, 0.65, 0.7))
    wᵢ(x, z) =   (- (x - 0.35) / σ^2) * vortex(x, z, 0.35, 0.7) + ((x - 0.65) / σ^2) * vortex(x, z, 0.65, 0.7)
    set!(model, u=(x, z) -> 0.1uᵢ(x, z), w=(x, z) -> 0.1wᵢ(x, z))

    u, v, w = model.velocities
    U² = Field(Integral(u^2))
    W² = Field(Integral(w^2))
    kinetic_energy() = (compute!(U²); compute!(W²); (only(Array(interior(U²))) + only(Array(interior(W²)))) / 2)

    Δt = cfl / N / maximum(abs, u)
    simulation = Simulation(model; Δt, stop_time, verbose=false)
    t, E = Float64[0], Float64[kinetic_energy()]
    record(sim) = (push!(t, time(sim)); push!(E, kinetic_energy()))
    add_callback!(simulation, record, IterationInterval(5))
    run!(simulation)

    return (; t, E)
end

function record_max_u!(simulation)
    u = simulation.model.velocities.u
    t, U = Float64[0], Float64[0]
    function record(sim)
        push!(t, time(sim)); push!(U, maximum(abs, u))
        (isfinite(U[end]) && U[end] < 1e3) || (sim.running = false)
    end
    add_callback!(simulation, record, IterationInterval(1))
    return t, U
end

# Cold blob sinking onto a staircase, in 2D.
function convection_case(Nx, Nz, immersed_boundary, z₀; stop_time=3)
    grid = RectilinearGrid(size=(Nx, Nz), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
    model = NonhydrostaticModel(ibg; advection=WENO(), tracers=:b, buoyancy=BuoyancyTracer(),
                                pressure_solver=ConjugateGradientPoissonSolver(ibg))
    set!(model, b = (x, z) -> - exp(-((x - 1/2)^2 + (z - z₀)^2) / (2 * 0.05^2)))
    simulation = Simulation(model; Δt=1e-3, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    t, U = record_max_u!(simulation)
    run!(simulation)
    return (; t, U)
end

# The 3D example from issue 6186.
function convection_3d_case(ImmersedBoundary; stop_time=5)
    grid = RectilinearGrid(size=(16, 16, 128), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Bounded, Bounded, Bounded))
    ibg = ImmersedBoundaryGrid(grid, ImmersedBoundary((x, y) -> staircase(x) + staircase(y)))
    model = NonhydrostaticModel(ibg; pressure_solver=ConjugateGradientPoissonSolver(ibg),
                                advection=WENO(), tracers=:b, buoyancy=BuoyancyTracer())
    set!(model, b = (x, y, z) -> -exp(-((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.55)^2) / (2 * 0.05^2)))
    simulation = Simulation(model; Δt=1e-3, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    t, U = record_max_u!(simulation)
    run!(simulation)
    return (; t, U)
end

cases = [
    "ke_partial"        => () -> kinetic_energy_case(PartialCellBottom(staircase)),
    "ke_partial_half_dt"=> () -> kinetic_energy_case(PartialCellBottom(staircase); cfl=0.1),
    "convection_test_grid_full"    => () -> convection_case(16, 128, GridFittedBottom(x -> 2staircase(x)), 0.55),
    "convection_128_full"    => () -> convection_case(128, 128, GridFittedBottom(staircase), 0.5; stop_time=6),
    "convection_3d_full"    => () -> convection_3d_case(GridFittedBottom),
]

for (name, case) in cases
    elapsed = @elapsed result = case()
    serialize(joinpath(outdir, name * ".jls"), result)
    @printf("%s done in %.0f s, final t = %.2f\n", name, elapsed, result.t[end])
    flush(stdout)
end
