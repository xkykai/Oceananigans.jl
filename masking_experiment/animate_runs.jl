using Oceananigans
using Oceananigans.Advection: ExplicitTimeDiscretization
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Serialization
using Printf

outdir, case = ARGS[1], ARGS[2]
mkpath(outdir)

staircase(x) = (5 + sum(tanh(40 * (x - n / 6)) for n in 1:5)) / 20

weno_with_second_order_advecting_velocity() =
    Oceananigans.Advection.WENO{3, Float64, Nothing}(nothing, WENO(order=3), Centered(order=2), ExplicitTimeDiscretization())

advection = case == "weno_c2" ? weno_with_second_order_advecting_velocity() : WENO()

function run_2d(advection; N=128, stop_time=6, frame_interval=0.05)
    grid = RectilinearGrid(size=(N, N), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg = ImmersedBoundaryGrid(grid, PartialCellBottom(staircase))
    model = NonhydrostaticModel(ibg; advection, tracers=:b, buoyancy=BuoyancyTracer(),
                                pressure_solver=ConjugateGradientPoissonSolver(ibg))
    set!(model, b = (x, z) -> - exp(-((x - 1/2)^2 + (z - 1/2)^2) / (2 * 0.05^2)))
    simulation = Simulation(model; Δt=1e-3, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)

    u, v, w = model.velocities
    b = model.tracers.b
    frames = []
    save_frame(sim) = push!(frames, (; t = time(sim),
                                       u = Array(interior(u, :, 1, :)), w = Array(interior(w, :, 1, :)),
                                       b = Array(interior(b, :, 1, :))))
    t, U = Float64[], Float64[]
    function record(sim)
        push!(t, time(sim)); push!(U, maximum(abs, u))
        (isfinite(U[end]) && U[end] < 1e3) || (sim.running = false)
    end
    add_callback!(simulation, save_frame, TimeInterval(frame_interval))
    add_callback!(simulation, record, IterationInterval(1))
    run!(simulation)

    return (; frames, t, U, x = collect(xnodes(ibg, Center())), z = collect(znodes(ibg, Center())),
              bottom = [staircase(x) for x in xnodes(ibg, Center())])
end

function run_3d(advection; stop_time=5)
    grid = RectilinearGrid(size=(16, 16, 128), x=(0, 1), y=(0, 1), z=(0, 1), topology=(Bounded, Bounded, Bounded))
    ibg = ImmersedBoundaryGrid(grid, PartialCellBottom((x, y) -> staircase(x) + staircase(y)))
    model = NonhydrostaticModel(ibg; pressure_solver=ConjugateGradientPoissonSolver(ibg),
                                advection, tracers=:b, buoyancy=BuoyancyTracer())
    set!(model, b = (x, y, z) -> -exp(-((x - 0.5)^2 + (y - 0.5)^2 + (z - 0.55)^2) / (2 * 0.05^2)))
    simulation = Simulation(model; Δt=1e-3, stop_time, verbose=false)
    conjure_time_step_wizard!(simulation, cfl=0.7)
    u = model.velocities.u
    t, U = Float64[], Float64[]
    function record(sim)
        push!(t, time(sim)); push!(U, maximum(abs, u))
        (isfinite(U[end]) && U[end] < 1e3) || (sim.running = false)
    end
    add_callback!(simulation, record, IterationInterval(1))
    run!(simulation)
    return (; t, U)
end

for (name, runner) in (("2d", run_2d), ("3d", run_3d))
    elapsed = @elapsed result = runner(advection)
    serialize(joinpath(outdir, "$(case)_$name.jls"), result)
    @printf("%s %s done in %.0f s, final t = %.2f, peak max|u| = %.3g\n", case, name, elapsed, result.t[end], maximum(filter(isfinite, result.U)))
    flush(stdout)
end
