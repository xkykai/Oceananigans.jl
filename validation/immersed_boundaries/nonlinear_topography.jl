using Oceananigans
using CairoMakie
using Printf
using Roots

include("immersed_pressure_solver.jl")

#####
##### Model setup
#####

function run_simulation(solver, preconditioner)
    Nz = 64
    Nx = Nz * 30
    Ny = 1
    
    grid = RectilinearGrid(GPU(), Float64,
                           size = (Nx, Ny, Nz), 
                           halo = (4, 4, 4),
                           x = (0, 30),
                           y = (0, 1),
                           z = (0, 1),
                           topology = (Periodic, Periodic, Bounded))
    

    k = 1
    Δt = 0.1e-3
    N² = 1 / (150 * 1e-3)^2
    U₀ = 1
    m = √(N² / U₀^2 - k^2)
    h₀ = 0.1

    function nonlinear_topography(h, x)
        return h₀ * cos(k*x + m*h) - h
    end

    topography(x, y) = find_zero(h -> nonlinear_topography(h, x), 1) + 0.1

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(topography))
    
    @info "Created $grid"
    
    uv_bcs = FieldBoundaryConditions(top=FluxBoundaryCondition(0), bottom=ValueBoundaryCondition(0), immersed=ValueBoundaryCondition(0))
    
    u_initial(x, y, z) = U₀ * z
    u_target = LinearTarget{:z}(intercept=0, gradient=U₀)
    
    b_initial(x, y, z) = N² * z
    b_target = LinearTarget{:z}(intercept=0, gradient=N²)
    
    mask = GaussianMask{:x}(center=28, width=0.5)
    damping_rate = 1 / (3 * Δt)
    v_sponge = w_sponge = Relaxation(rate=damping_rate, mask=mask)
    u_sponge = Relaxation(rate=damping_rate, mask=mask, target=u_target)
    b_sponge = Relaxation(rate=damping_rate, mask=mask, target=b_target)
    
    if solver == "FFT"
        model = NonhydrostaticModel(; grid,
                                    advection = WENO(),
                                    coriolis = FPlane(f=0.1),
                                    tracers = (:b, :c),
                                    buoyancy = BuoyancyTracer(),
                                    timestepper = :RungeKutta3,
                                    boundary_conditions=(; u=uv_bcs, v=uv_bcs),
                                    forcing=(u=u_sponge, v=v_sponge, w=w_sponge, b=b_sponge))
    else
        model = NonhydrostaticModel(; grid,
                                    pressure_solver = ImmersedPoissonSolver(grid, preconditioner=preconditioner, reltol=1e-8),
                                    advection = WENO(),
                                    coriolis = FPlane(f=0.1),
                                    tracers = (:b, :c),
                                    buoyancy = BuoyancyTracer(),
                                    timestepper = :RungeKutta3,
                                    boundary_conditions=(; u=uv_bcs, v=uv_bcs),
                                    forcing=(u=u_sponge, v=v_sponge, w=w_sponge, b=b_sponge))
    end

    @info "Created $model"
    @info "with pressure solver $(model.pressure_solver)"
    @info model.velocities.u.boundary_conditions
    
    set!(model, b=b_initial, c=1, u=u_initial)
    
    #####
    ##### Simulation
    #####
    
    simulation = Simulation(model, Δt=Δt, stop_time=20)

    wizard = TimeStepWizard(max_change=1.05, max_Δt=1e-3, cfl=0.6)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))

    wall_time = Ref(time_ns())
    
    b, c = model.tracers
    u, v, w = model.velocities
    B = Field(Integral(b))
    C = Field(Integral(c))
    compute!(B)
    compute!(C)
    
    δ = Field(∂x(u) + ∂y(v) + ∂z(w))
    compute!(δ)
    
    ζ = Field(∂z(u) - ∂x(w))
    compute!(ζ)
    
    function print_progress(sim)
        elapsed = time_ns() - wall_time[]

        msg = @sprintf("[%05.2f%%] i: %d, t: %s, wall time: %s, max(u): (%6.3e, %6.3e, %6.3e) m/s, max(b) %6.3e, next Δt: %s",
                        100 * (sim.model.clock.time / sim.stop_time),
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(1e-9 * elapsed),
                        maximum(abs, sim.model.velocities.u),
                        maximum(abs, sim.model.velocities.v),
                        maximum(abs, sim.model.velocities.w),
                        maximum(abs, sim.model.tracers.b),
                        prettytime(sim.Δt))

        pressure_solver = sim.model.pressure_solver
        if sim.model.pressure_solver isa ImmersedPoissonSolver
            solver_iterations = pressure_solver.pcg_solver.iteration 
            msg *= string(", solver iterations: ", solver_iterations)
        end
    
        @info msg
    
        wall_time[] = time_ns()
    
        return nothing
    end
                       
    simulation.callbacks[:p] = Callback(print_progress, IterationInterval(1))
    
    solver_type = model.pressure_solver isa ImmersedPoissonSolver ? "ImmersedPoissonSolver" : "FFTBasedPoissonSolver"
    prefix = "nonlinear_topography_" * solver_type
    
    outputs = merge(model.velocities, model.tracers, (; p=model.pressures.pNHS, δ, ζ))
    
    simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                        filename = prefix * "_fieldss",
                                                        schedule = TimeInterval(0.01),
                                                        # schedule = IterationInterval(1),
                                                        overwrite_existing = true)
    
    simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; B, C);
                                                              filename = prefix * "_time_seriess",
                                                              schedule = TimeInterval(0.01),
                                                        # schedule = IterationInterval(1),
                                                              overwrite_existing = true)
    
    run!(simulation)
end

run_simulation("ImmersedPoissonSolver", "FFT")
run_simulation("FFT", nothing)

# function nonlinear_topography(h, x)
#     k = 1
#     N² = 1 / (150 * 0.5e-2)^2
#     U₀ = 1
#     m = √(N² / U₀^2 - k^2)
#     h₀ = 0.1
#     return h₀ * cos(k*x + m*h) - h
# end

# xs = 0:0.1:100π
# hs = [find_zero(h -> nonlinear_topography(h, x), 1) for x in xs]

# fig = Figure()
# ax = Axis(fig[1,1])
# lines!(ax, xs, hs)
# lines!(ax, xs, cos.(xs) .* 0.1)
# display(fig)

#####
##### Visualize
#####
##
filename_FFT = "nonlinear_topography_FFTBasedPoissonSolver_fieldss.jld2"
bt_FFT = FieldTimeSeries(filename_FFT, "b")
ut_FFT = FieldTimeSeries(filename_FFT, "u")
wt_FFT = FieldTimeSeries(filename_FFT, "w")
δt_FFT = FieldTimeSeries(filename_FFT, "δ")
times = bt_FFT.times

filename_PCG = "nonlinear_topography_ImmersedPoissonSolver_fieldss.jld2"
bt_PCG = FieldTimeSeries(filename_PCG, "b")
ut_PCG = FieldTimeSeries(filename_PCG, "u")
wt_PCG = FieldTimeSeries(filename_PCG, "w")
δt_PCG = FieldTimeSeries(filename_PCG, "δ")

fig = Figure(resolution=(2000, 700))
n = Observable(1)

titlestr = @lift @sprintf("t = %.2f", times[$n])

axb_FFT = Axis(fig[1, 1], title="b (FFT solver)")
axu_FFT = Axis(fig[1, 2], title="u (FFT solver)")
axw_FFT = Axis(fig[1, 3], title="w (FFT solver)")
axd_FFT = Axis(fig[1, 4], title="Divergence (FFT solver)")

axb_PCG = Axis(fig[2, 1], title="b (PCG solver)")
axu_PCG = Axis(fig[2, 2], title="u (PCG solver)")
axw_PCG = Axis(fig[2, 3], title="w (PCG solver)")
axd_PCG = Axis(fig[2, 4], title="Divergence (PCG solver)")

bn_FFT = @lift interior(bt_FFT[$n], :, 1, :)
un_FFT = @lift interior(ut_FFT[$n], :, 1, :)
wn_FFT = @lift interior(wt_FFT[$n], :, 1, :)
δn_FFT = @lift interior(δt_FFT[$n], :, 1, :)

bn_PCG = @lift interior(bt_PCG[$n], :, 1, :)
un_PCG = @lift interior(ut_PCG[$n], :, 1, :)
wn_PCG = @lift interior(wt_PCG[$n], :, 1, :)
δn_PCG = @lift interior(δt_PCG[$n], :, 1, :)

Nx = bt_FFT.grid.Nx
Nz = bt_FFT.grid.Nz
Nt = length(bt_FFT.times)

xC = bt_FFT.grid.xᶜᵃᵃ[1:Nx]
zC = bt_FFT.grid.zᵃᵃᶜ[1:Nz]

blim = maximum([maximum(abs, bt_FFT), maximum(abs, bt_PCG)])
ulim = maximum([maximum(abs, ut_FFT), maximum(abs, ut_PCG)])
wlim = maximum([maximum(abs, wt_FFT), maximum(abs, wt_PCG)])
δlim = 1e-8

heatmap!(axb_FFT, xC, zC, bn_FFT, colormap=:balance, colorrange=(0, blim))
heatmap!(axu_FFT, xC, zC, un_FFT, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axw_FFT, xC, zC, wn_FFT, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axd_FFT, xC, zC, δn_FFT, colormap=:balance, colorrange=(-δlim, δlim))

heatmap!(axb_PCG, xC, zC, bn_PCG, colormap=:balance, colorrange=(0, blim))
heatmap!(axu_PCG, xC, zC, un_PCG, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axw_PCG, xC, zC, wn_PCG, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axd_PCG, xC, zC, δn_PCG, colormap=:balance, colorrange=(-δlim, δlim))

Label(fig[0, :], titlestr, font=:bold, tellwidth=false, tellheight=false)

# display(fig)

record(fig, "FFT_PCG_nonlinear_topography.mp4", 1:Nt, framerate=30) do nn
    # @info string("Plotting frame ", nn, " of ", Nt)
    n[] = nn
end
## 