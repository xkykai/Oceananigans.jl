using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, znode
using Serialization
using Printf

# Lee wave over a Gaussian bump from issue 5693
outdir, scheme_name = ARGS[1], ARGS[2]
mkpath(outdir)

Nx, Nz = 100, 100
H, L = 2kilometers, 20kilometers
h₀, width, x₀ = 180meters, 2kilometers, -10kilometers
bottom(x) = - H + h₀ * exp(-(x - x₀)^2 / 2width^2)
f, U, N² = 1e-4, 0.2, 1e-6

underlying_grid = RectilinearGrid(size=(Nx, Nz), halo=(7, 7), x=(-L, L), z=(-H, 0), topology=(Periodic, Flat, Bounded))
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

sponge_mask(x) = 1 + (tanh((x - 4L/5) / (L/5)) - tanh((x + 4L/5) / (L/5))) / 2
t_restore = 4hours
@inline u_sponge(i, j, k, grid, clock, fields, p) = - sponge_mask(xnode(i, j, k, grid, Face(), Center(), Center())) * (@inbounds fields.u[i, j, k] - p.U) / p.t_restore
@inline v_sponge(i, j, k, grid, clock, fields, p) = - sponge_mask(xnode(i, j, k, grid, Center(), Face(), Center())) * @inbounds(fields.v[i, j, k]) / p.t_restore
@inline b_sponge(i, j, k, grid, clock, fields, p) = - sponge_mask(xnode(i, j, k, grid, Center(), Center(), Center())) *
                                                     (@inbounds fields.b[i, j, k] - p.N² * znode(i, j, k, grid, Center(), Center(), Center())) / p.t_restore

parameters = (; U, N², t_restore)
forcing = (u = Forcing(u_sponge; discrete_form=true, parameters),
           v = (Forcing((x, z, t) -> f * U), Forcing(v_sponge; discrete_form=true, parameters)),
           b = Forcing(b_sponge; discrete_form=true, parameters))

momentum_advection = scheme_name == "WENO" ? WENO() :
                     scheme_name == "WENOVectorInvariant" ? WENOVectorInvariant() : VectorInvariant()

model = HydrostaticFreeSurfaceModel(grid; coriolis=FPlane(; f), buoyancy=BuoyancyTracer(), tracers=:b,
                                    tracer_advection=WENO(), momentum_advection, forcing,
                                    closure=ScalarDiffusivity(ν=0.1, κ=0.1))
set!(model, u=U, b=(x, z) -> N² * z)

simulation = Simulation(model, Δt=20seconds, stop_time=15days, verbose=false)
conjure_time_step_wizard!(simulation, IterationInterval(100), cfl=0.2, max_Δt=2minutes)
elapsed = @elapsed run!(simulation)

u, v, w = model.velocities
result = (; u = Array(interior(u, :, 1, :)), w = Array(interior(w, :, 1, :)),
            x = collect(xnodes(grid, Face())), z = collect(znodes(grid, Center())))
serialize(joinpath(outdir, "lee_$scheme_name.jls"), result)
@printf("%s done in %.0f s, max|u - U| = %.4f, max|w| = %.2e\n", scheme_name, elapsed, maximum(abs, result.u .- U), maximum(abs, result.w))
