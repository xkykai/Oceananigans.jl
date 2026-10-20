using Oceananigans
using Printf

# Two horizontal boxes separated by a one-cell wall, hydrostatic, vector invariant momentum advection
function right_box_after(momentum_advection, left_vortex)
    grid = RectilinearGrid(size=(33, 16, 4), x=(0, 2 + 1/16), y=(0, 1), z=(0, 1), halo=(7, 7, 4),
                           topology=(Bounded, Bounded, Bounded))
    grid = ImmersedBoundaryGrid(grid, GridFittedBottom((x, y) -> 1 < x < 1 + 1/16 ? 1 : 0)) # one-cell wall, full depth
    model = HydrostaticFreeSurfaceModel(grid; momentum_advection, free_surface=ImplicitFreeSurface())

    A(x) = x < 1 ? left_vortex : x > 1 + 1/16 ? 1 : 0
    ξ(x) = x < 1 ? x : x - 1 - 1/16
    set!(model, u = (x, y, z) ->  2π * A(x) * sin(π * ξ(x))^2 * sin(π * y) * cos(π * y),
                v = (x, y, z) -> -2π * A(x) * sin(π * ξ(x)) * cos(π * ξ(x)) * sin(π * y)^2)

    run!(Simulation(model; Δt=1e-3, stop_time=1, verbose=false))
    return Array(interior(model.velocities.v, :, :, 1))
end

using Serialization
advection = VectorInvariant(vorticity_scheme=WENO(order=9))
serialize(ARGS[1], (; vᴬ = right_box_after(advection, 1), vᴮ = right_box_after(advection, 0)))
