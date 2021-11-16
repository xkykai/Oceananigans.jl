using Oceananigans
using Oceananigans.Grids: min_Δx, min_Δy, min_Δz
using JLD2
using OffsetArrays
using BenchmarkTools
using Test
using LinearAlgebra
using Plots
"""

This simulation is a simple 1D advection of a gaussian function, to test the 
validity of the stretched WENO scheme
    
"""
N    = 32
arch = CPU()

# regular "stretched" mesh
Freg = range(0,1,length = N+1)

# seasaw mesh
Fsaw(j) = 1 / N  * (j - 1) + 0.1 * 1 / N * mod(j - 1, 2)
  
function Δstr2(i, N)
    if i < N/4
     return 1
    elseif i > N/4*3
     return 1
    elseif i<N/2
     return 1.2 * (i - N/4) + 1
    else
     return  1.2 * (3*N/4 - i) + 1
    end
end   

 Fstr2 = zeros(Float64, N+1)

 for i = 2:N+1
     Fstr2[i] = Fstr2[i-1] + Δstr2(i-1, N)
 end

 Fstr2 ./= Fstr2[end]

solution  = Dict()
real_sol  = Dict()
coord     = Dict()
residual  = Dict()

grid_reg  = RectilinearGrid(size = (N,), x = Freg,  halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
grid_str  = RectilinearGrid(size = (N,), x = Fsaw,  halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    
grid_str2 = RectilinearGrid(size = (N,), x = Fstr2, halo = (3,), topology = (Periodic, Flat, Flat), architecture = arch)    

advection = [WENO5(), WENO5()]

schemes = [:center4, :wreg, :wstr]

# Checking the accuracy of different schemes with different settings

# for grid in [grid_reg, grid_str, grid_str2], (adv, scheme) in enumerate(advection) 


#     grid == grid_reg ? gr = :reg : grid == grid_str ? gr = :str : gr = :str2

#     U = Field(Face, Center, Center, arch, grid)
#     parent(U) .= 1 

#     if adv == 2
#         scheme = WENO5(grid = grid)
#     end

#     model = HydrostaticFreeSurfaceModel(architecture = arch,
#                                                 grid = grid,
#                                              tracers = :c,
#                                     tracer_advection = scheme,
#                                           velocities = PrescribedVelocityFields(u=U), 
#                                             coriolis = nothing,
#                                              closure = nothing,
#                                             buoyancy = nothing)

    
#     Δt_max   = 0.2 * min_Δx(grid)
#     end_time = 1000 * Δt_max
#     c₀(x, y, z) = 10*exp(-((x-0.5)/0.1)^2)
                                            
#     set!(model, c=c₀)
#     c = model.tracers.c

#     x        = grid.xᶜᵃᵃ[1:grid.Nx]
#     creal    = Array(c₀.(mod.((x .- end_time),1), 0, 0))

#     simulation = Simulation(model,
#                             Δt = Δt_max,
#                             stop_time = end_time)                           
#     run!(simulation, pickup=false)

#     ctest   = Array(parent(c.data))
#     offsets = (c.data.offsets[1],  c.data.offsets[2],  c.data.offsets[3])
#     ctemp   = OffsetArray(ctest, offsets)
#     ctests  = ctemp[1:grid.Nx, 1, 1]

#     real_sol[(gr)] = creal
#     coord[(gr)]    = x
    
#     residual[(schemes[adv], gr)] = norm(abs.(creal .- ctests))
#     solution[(schemes[adv], gr)] = ctests
# end

"""
Now test a 2D simulation (to do)
"""

solution2D  = Dict()
real_sol2D  = Dict()
coord2D     = Dict()
residual2D  = Dict()

grid_reg  = RectilinearGrid(size = (N, N), x = Freg,  y = Freg,  halo = (4, 4), topology = (Periodic, Flat, Periodic), architecture = arch)    
grid_str  = RectilinearGrid(size = (N, N), x = Fsaw,  y = Fsaw,  halo = (4, 4), topology = (Periodic, Flat, Periodic), architecture = arch)    
grid_str2 = RectilinearGrid(size = (N, N), x = Fstr2, y = Fstr2, halo = (4, 4), topology = (Periodic, Flat, Periodic), architecture = arch)    

for grid in [grid_reg, grid_str, grid_str2]
    
    grid == grid_reg ? gr = :reg : grid == grid_str ? gr = :str : gr = :str2

    U = Field(Face, Center, Center, arch, grid); set!(U, (x, y, z) -> - 2y + 1)
    V = Field(Center, Face, Center, arch, grid); set!(V, (x, y, z) ->   2x - 1)
    # U = Field(Center, Face, Center, arch, grid); set!(U, (x, y, z) -> - 2z + 1)
    # V = Field(Center, Center, Face, arch, grid); set!(V, (x, y, z) ->   2y - 1)
    # U = Field(Face, Center, Center, arch, grid); set!(U, (x, y, z) -> - 2z + 1)
    # V = Field(Center, Center, Face, arch, grid); set!(V, (x, y, z) ->   2x - 1)

    Δt_max   = 0.1 * min_Δx(grid)
    end_time = 3000 * Δt_max

    for (adv, scheme) in enumerate(advection) 

        if adv == 2
            scheme = WENO5(grid = grid)
        end

        model = HydrostaticFreeSurfaceModel(architecture = arch,
                                                    grid = grid,
                                                tracers = :c,
                                        tracer_advection = scheme,
                                            velocities = PrescribedVelocityFields(u=U, v=V), 
                                                coriolis = nothing,
                                                closure  = nothing,
                                                buoyancy = nothing)


        mask(y) = y < 0.75 && y > 0.15 ? 1 : 0
        c₀(x, y, z) = 10*exp(-((x-0.5)/0.3)^2) * mask(y)
        # c₀(x, y, z) = 10*exp(-((x-0.5)/0.3)^2) * mask(z)
        # c₀(x, y, z) = 10*exp(-((y-0.5)/0.3)^2) * mask(z)
                                                
        set!(model, c=c₀)
        c = model.tracers.c

        simulation = Simulation(model,
                                Δt = Δt_max,
                                stop_time = end_time)                           

        for i = 1:end_time/Δt_max/10
            for j = 1:10
                time_step!(model, Δt_max)
            end
            ctest   = Array(parent(model.tracers.c.data))
            offsets = (model.tracers.c.data.offsets[1],  model.tracers.c.data.offsets[2],  model.tracers.c.data.offsets[3])
            ctemp   = OffsetArray(ctest, offsets)
            # ctests  = ctemp[1 ,1:grid.Ny, 1:grid.Nz]
            # ctests  = ctemp[1:grid.Nx, 1 ,1:grid.Nz]
            ctests  = ctemp[1:grid.Nx, 1:grid.Ny, 1]
            solution2D[(schemes[adv], gr, Int(i))] = ctests
        end

    end

    x        = grid.xᶜᵃᵃ[1:grid.Nx]
    y        = grid.yᵃᶜᵃ[1:grid.Ny]
    # y        = grid.zᵃᵃᶜ[1:grid.Nz]
    coord2D[(gr)]    = (x, y)
    anim = @animate for i ∈ 1:end_time/Δt_max/10
        plot(contourf(x, y, solution2D[(schemes[1], gr, Int(i))], clim=(-1, 11), levels = -1:1:11),
             contourf(x, y, solution2D[(schemes[2], gr, Int(i))], clim=(-1, 11), levels = -1:1:11))
    end 
    gif(anim, "anim_$gr.mp4", fps = 15)
end

