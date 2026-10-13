using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Serialization
using Printf

outdir = ARGS[1]
mkpath(outdir)

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20

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
function convection_case(Nx, Nz, immersed_boundary, z₀; stop_time=6, snapshot_times=())
    grid = RectilinearGrid(size=(Nx, Nz), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg = ImmersedBoundaryGrid(grid, immersed_boundary)
    model = NonhydrostaticModel(ibg; advection=WENO(), tracers=:b, buoyancy=BuoyancyTracer(),
                                pressure_solver=ConjugateGradientPoissonSolver(ibg))
    set!(model, b = (x, z) -> - exp(-((x - 1/2)^2 + (z - z₀)^2) / (2 * 0.05^2)))
    simulation = Simulation(model; Δt=1e-3, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    t, U = record_max_u!(simulation)

    u, v, w = model.velocities
    b = model.tracers.b
    snapshots = []
    for T in (snapshot_times..., stop_time)
        simulation.stop_time = T
        run!(simulation)
        T in snapshot_times && push!(snapshots, (; t=time(simulation), u=Array(interior(u, :, 1, :)), b=Array(interior(b, :, 1, :))))
        current = maximum(abs, u)
        (isfinite(current) && current < 1e3) || break
    end
    return (; t, U, snapshots)
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

all_cases = Dict(
    "conv_128_partial"  => () -> convection_case(128, 128, PartialCellBottom(staircase), 0.5; snapshot_times=(2, 4, 6)),
    "conv_128_full"     => () -> convection_case(128, 128, GridFittedBottom(staircase), 0.5; snapshot_times=(2, 4, 6)),
    "conv_16_partial"   => () -> convection_case(16, 128, PartialCellBottom(x -> 2staircase(x)), 0.55; stop_time=3),
    "conv_16_full"      => () -> convection_case(16, 128, GridFittedBottom(x -> 2staircase(x)), 0.55; stop_time=3),
    "conv_3d_partial"   => () -> convection_3d_case(PartialCellBottom),
    "conv_3d_full"      => () -> convection_3d_case(GridFittedBottom),
)

for name in ARGS[2:end]
    elapsed = @elapsed result = all_cases[name]()
    serialize(joinpath(outdir, name * ".jls"), result)
    @printf("%s done in %.0f s, final t = %.2f, peak max|u| = %.3g\n", name, elapsed, result.t[end], maximum(filter(isfinite, result.U)))
    flush(stdout)
end
