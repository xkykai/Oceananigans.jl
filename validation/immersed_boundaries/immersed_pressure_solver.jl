using Oceananigans
using Oceananigans.Operators

using Oceananigans.Architectures: device, device_event, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, FFTBasedPoissonSolver, FourierTridiagonalPoissonSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Grids: inactive_cell
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Utils: launch!
using Oceananigans.Models.NonhydrostaticModels: PressureSolver, calculate_pressure_source_term_fft_based_solver!
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: precondition!
import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

struct ImmersedPoissonSolver{R, G, S}
    rhs :: R
    grid :: G
    pcg_solver :: S
end

@kernel function fft_preconditioner_right_hand_side!(preconditioner_rhs, rhs)
    i, j, k = @index(Global, NTuple)
    @inbounds preconditioner_rhs[i, j, k] = rhs[i, j, k]
end

# FFTBasedPoissonPreconditioner
function precondition!(p, solver::FFTBasedPoissonSolver, rhs, args...)
    grid = solver.grid
    arch = architecture(grid)

    event = launch!(arch, grid, :xyz,
                    fft_preconditioner_right_hand_side!,
                    solver.storage, rhs,
                    dependencies = device_event(arch))

    wait(device(arch), event)

    return solve!(p, solver, solver.storage)
end

function ImmersedPoissonSolver(grid;
                               preconditioner = true,
                               reltol = eps(eltype(grid)),
                               abstol = 0,
                               kw...)

    if preconditioner
        arch = architecture(grid)
        preconditioner = PressureSolver(arch, grid)
    else
        preconditioner = nothing
    end

    rhs = CenterField(grid)

    pcg_solver = PreconditionedConjugateGradientSolver(compute_laplacian!; reltol, abstol,
                                                       preconditioner,
                                                       template_field = rhs,
                                                       kw...)

    return ImmersedPoissonSolver(rhs, grid, pcg_solver)
end

@kernel function calculate_pressure_source_term!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@inline laplacianᶜᶜᶜ(i, j, k, grid, ϕ) = ∇²ᶜᶜᶜ(i, j, k, grid, ϕ)

@kernel function laplacian!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = laplacianᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_laplacian!(∇²ϕ, ϕ)
    grid = ϕ.grid
    arch = architecture(grid)

    fill_halo_regions!(ϕ)

    event = launch!(arch, grid, :xyz, laplacian!, ∇²ϕ, grid, ϕ, dependencies=device_event(arch))
    wait(device(arch), event)

    return nothing
end

function solve_for_pressure!(pressure, solver::ImmersedPoissonSolver, Δt, U★)
    rhs = solver.rhs
    grid = solver.grid
    arch = architecture(grid)

    if grid isa ImmersedBoundaryGrid
        underlying_grid = grid.underlying_grid
    else
        underlying_grid = grid
    end

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term!,
                        rhs, underlying_grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    event = mask_immersed_field!(rhs, zero(grid))
    wait(device(arch), event)

    # Solve pressure Pressure equation for pressure, given rhs
    solve!(pressure, solver.pcg_solver, rhs)

    return pressure
end
