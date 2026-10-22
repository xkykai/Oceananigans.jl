using Oceananigans
using Oceananigans.Units
using Oceananigans.Grids: xnode, znode
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Serialization
using Printf

# Lee wave over a Gaussian bump from issue 5693
outdir, scheme_name, N = ARGS[1], ARGS[2], parse(Int, ARGS[3])
mkpath(outdir)

Nx, Nz = N, N
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

model = NonhydrostaticModel(grid; coriolis=FPlane(; f), buoyancy=BuoyancyTracer(), tracers=:b, advection=WENO(), forcing,
                            closure=ScalarDiffusivity(VerticallyImplicitTimeDiscretization(); ν=0.1, κ=0.1),
                            pressure_solver=ConjugateGradientPoissonSolver(grid))
set!(model, u=U, b=(x, z) -> N² * z)

simulation = Simulation(model, Δt=20seconds * 100 / N, stop_time=15days, verbose=false)
conjure_time_step_wizard!(simulation, IterationInterval(500), cfl=0.2, max_Δt=2minutes * 100 / N)
elapsed = @elapsed run!(simulation)

u, v, w = model.velocities
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: Vᶜᶜᶜ
volume = [inactive_cell(i, 1, k, grid) ? 0.0 : Vᶜᶜᶜ(i, 1, k, grid) for i in 1:Nx, k in 1:Nz]
result = (; u = Array(interior(u, :, 1, :)), w = Array(interior(w, :, 1, :)), b = Array(interior(model.tracers.b, :, 1, :)),
            x = collect(xnodes(grid, Face())), z = collect(znodes(grid, Center())), volume, N)
serialize(joinpath(outdir, "lee_nonhydrostatic_$N.jls"), result)
@printf("nonhydrostatic %s N=%d done in %.0f s, max|u - U| = %.4f, max|w| = %.2e\n", scheme_name, N, elapsed, maximum(abs, result.u .- U), maximum(abs, result.w))
