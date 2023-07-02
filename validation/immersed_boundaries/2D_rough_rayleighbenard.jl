using Oceananigans
using CairoMakie
using Printf

include("immersed_pressure_solver.jl")

#####
##### Model setup
#####

function run_simulation(solver, preconditioner; Nr, Ra, Pr=1)
    Lx = 1
    Ly = 1
    Lz = 1

    h = Lx / Nr / 2
    x₀s = h:2h:Lx-h

    ν = 1
    κ = ν / Pr

    S = Ra * ν * κ / Lz ^ 4

    Nz = 256
    Nx = Nz
    Ny = 1
    
    grid = RectilinearGrid(GPU(), Float64,
                           size = (Nx, Ny, Nz), 
                           halo = (4, 4, 4),
                           x = (0, Lx),
                           y = (0, Ly),
                           z = (0, Lz),
                           topology = (Bounded, Periodic, Bounded))

    @inline function local_roughness(x, x₀, h)
        if x > x₀ - h && x <= x₀
            return x + h - x₀
        elseif x > x₀ && x <= x₀ + h
            return -x + h + x₀
        else
            return 0
        end
    end

    topography(x, y) = sum([local_roughness(x, x₀, h) for x₀ in x₀s])

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(topography))
    
    @info "Created $grid"
    
    # uvw_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0), bottom=ValueBoundaryCondition(0),
    #                                   west=ValueBoundaryCondition(0), east=ValueBoundaryCondition(0))
    # uvw_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0), immersed=ValueBoundaryCondition(0))
    u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0), bottom=ValueBoundaryCondition(0), immersed=ValueBoundaryCondition(0))
    v_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0), bottom=ValueBoundaryCondition(0),
                                    east=ValueBoundaryCondition(0), west=ValueBoundaryCondition(0),
                                    immersed=ValueBoundaryCondition(0))
    w_bcs = FieldBoundaryConditions(east=ValueBoundaryCondition(0), west=ValueBoundaryCondition(0),
                                    immersed=ValueBoundaryCondition(0))

    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(-S/2), bottom=ValueBoundaryCondition(S/2), 
                                    west=FluxBoundaryCondition(0), east=FluxBoundaryCondition(0),
                                    immersed=ValueBoundaryCondition(S/2))
    
    # Δt = 1e-5 / 4
    Δt = 5e-6
    max_Δt = 1e-4
    
    if solver == "FFT"
        model = NonhydrostaticModel(; grid,
                                    advection = WENO(),
                                    tracers = (:b),
                                    buoyancy = BuoyancyTracer(),
                                    # timestepper = :RungeKutta3,
                                    boundary_conditions=(; u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs))
    else
        model = NonhydrostaticModel(; grid,
                                    pressure_solver = ImmersedPoissonSolver(grid, preconditioner=preconditioner, reltol=1e-8),
                                    advection = WENO(),
                                    tracers = (:b),
                                    buoyancy = BuoyancyTracer(),
                                    # timestepper = :RungeKutta3,
                                    boundary_conditions=(; u=u_bcs, v=v_bcs, w=w_bcs, b=b_bcs))
    end

    @info "Created $model"
    @info "with pressure solver $(model.pressure_solver)"
    @info "with b boundary conditions $(model.tracers.b.boundary_conditions)"
    @info "u, v, w boundary conditions $(model.velocities.u.boundary_conditions)"

    b_initial(x, y, z) = -S * z + S / 2 - rand() * Ra / 100000
    
    set!(model, b=b_initial)
    
    #####
    ##### Simulation
    #####
    
    simulation = Simulation(model, Δt=Δt, stop_time=1)

    # wizard = TimeStepWizard(max_change=1.05, max_Δt=max_Δt, cfl=0.6)
    # simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(1))
    
    wall_time = Ref(time_ns())
    
    b = model.tracers.b
    u, v, w = model.velocities
    WB = Average(w * b, dims=(1, 2))
    
    δ = Field(∂x(u) + ∂y(v) + ∂z(w))
    compute!(δ)

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
    prefix = "2D_rough_rayleighbenard_" * solver_type
    
    outputs = merge(model.velocities, model.tracers, (; δ))
    
    simulation.output_writers[:jld2] = JLD2OutputWriter(model, outputs;
                                                        filename = prefix * "_fields",
                                                        schedule = TimeInterval(0.01),
                                                        # schedule = IterationInterval(1),
                                                        overwrite_existing = true)
    
    simulation.output_writers[:timeseries] = JLD2OutputWriter(model, (; WB);
                                                              filename = prefix * "_time_series",
                                                              schedule = TimeInterval(0.01),
                                                        # schedule = IterationInterval(1),
                                                              overwrite_existing = true)
    
    run!(simulation)
end

Nr = 2
Ra = 1e5

run_simulation("ImmersedPoissonSolver", "FFT", Nr=Nr, Ra=Ra)
run_simulation("FFT", nothing, Nr=Nr, Ra=Ra)

#####
##### Visualize
#####
##
filename_FFT = "2D_rough_rayleighbenard_FFTBasedPoissonSolver_fields.jld2"
bt_FFT = FieldTimeSeries(filename_FFT, "b")
ut_FFT = FieldTimeSeries(filename_FFT, "u")
wt_FFT = FieldTimeSeries(filename_FFT, "w")
δt_FFT = FieldTimeSeries(filename_FFT, "δ")
times = bt_FFT.times

filename_PCG = "2D_rough_rayleighbenard_ImmersedPoissonSolver_fields.jld2"
bt_PCG = FieldTimeSeries(filename_PCG, "b")
ut_PCG = FieldTimeSeries(filename_PCG, "u")
wt_PCG = FieldTimeSeries(filename_PCG, "w")
δt_PCG = FieldTimeSeries(filename_PCG, "δ")

fig = Figure(resolution=(1500, 1000))
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

heatmap!(axb_FFT, xC, zC, bn_FFT, colormap=:balance, colorrange=(-blim, blim))
heatmap!(axu_FFT, xC, zC, un_FFT, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axw_FFT, xC, zC, wn_FFT, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axd_FFT, xC, zC, δn_FFT, colormap=:balance, colorrange=(-δlim, δlim))

heatmap!(axb_PCG, xC, zC, bn_PCG, colormap=:balance, colorrange=(-blim, blim))
heatmap!(axu_PCG, xC, zC, un_PCG, colormap=:balance, colorrange=(-ulim, ulim))
heatmap!(axw_PCG, xC, zC, wn_PCG, colormap=:balance, colorrange=(-wlim, wlim))
heatmap!(axd_PCG, xC, zC, δn_PCG, colormap=:balance, colorrange=(-δlim, δlim))

Label(fig[0, :], titlestr, font=:bold, tellwidth=false, tellheight=false)

# display(fig)

record(fig, "FFT_PCG_2D_rough_rayleighbenard.mp4", 1:Nt, framerate=30) do nn
    # @info string("Plotting frame ", nn, " of ", Nt)
    n[] = nn
end
## 