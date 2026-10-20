using Oceananigans
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Serialization

function u_after(left_vortex)
    grid = RectilinearGrid(size=(33, 16), x=(0, 2 + 1/16), z=(0, 1), topology=(Bounded, Flat, Bounded))
    grid = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, z) -> 1 < x < 1 + 1/16))
    model = NonhydrostaticModel(grid; advection=WENO(), pressure_solver=ConjugateGradientPoissonSolver(grid))
    A(x) = x < 1 ? left_vortex : x > 1 + 1/16 ? 1 : 0
    ξ(x) = x < 1 ? x : x - 1 - 1/16
    set!(model, u = (x, z) ->  2π * A(x) * sin(π * ξ(x))^2 * sin(π * z) * cos(π * z),
                w = (x, z) -> -2π * A(x) * sin(π * ξ(x)) * cos(π * ξ(x)) * sin(π * z)^2)
    run!(Simulation(model; Δt=1e-3, stop_time=1, verbose=false))
    return Array(interior(model.velocities.u, :, 1, :))
end

serialize(ARGS[1], (; uᴬ = u_after(1), uᴮ = u_after(0)))
