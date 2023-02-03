using Oceananigans
using Oceananigans.Operators
using Oceananigans.Architectures: device, device_event, architecture
using Oceananigans.Solvers: PreconditionedConjugateGradientSolver, solve!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: divᶜᶜᶜ
using Oceananigans.Utils: launch!
using KernelAbstractions: @kernel, @index

import Oceananigans.Models.NonhydrostaticModels: solve_for_pressure!

# const ZXYPermutation = Permutation{(3, 1, 2), 3}
# const ZYXPermutation = Permutation{(3, 2, 1), 3}

struct ImmersedPoissonSolver{R, G, S}
    rhs :: R
    grid :: G
    pcg_solver :: S
    # permutation :: P
end

function ImmersedPoissonSolver(grid; reltol = eps(eltype(grid)), kw...)
    rhs = CenterField(grid)

    pcg_solver = PreconditionedConjugateGradientSolver(compute_laplacian!;
                                                       template_field = rhs,
                                                       reltol,
                                                       kw...)

    return ImmersedPoissonSolver(rhs, grid, pcg_solver)
end

@kernel function calculate_pressure_source_term!(rhs, grid, Δt, U★)
    i, j, k = @index(Global, NTuple)
    @inbounds rhs[i, j, k] = divᶜᶜᶜ(i, j, k, grid, U★.u, U★.v, U★.w) / Δt
end

@inline laplacianᶜᶜᶜ(i, j, k, grid, ϕ) = ∂xᶜᶜᶜ(i, j, k, grid, ∂xᶠᶜᶜ, ϕ) +
                                         ∂yᶜᶜᶜ(i, j, k, grid, ∂yᶜᶠᶜ, ϕ) +
                                         ∂zᶜᶜᶜ(i, j, k, grid, ∂zᶜᶜᶠ, ϕ)

@kernel function laplacian!(∇²ϕ, grid, ϕ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ϕ[i, j, k] = laplacianᶜᶜᶜ(i, j, k, grid, ϕ)
end

function compute_laplacian!(∇²ϕ, ϕ)
    fill_halo_regions!(ϕ)
    grid = ϕ.grid
    arch = architecture(grid)
    event = launch!(arch, grid, :xyz, laplacian!, ∇²ϕ, grid, ϕ, dependencies=device_event(arch))
    wait(device(arch), event)
    fill_halo_regions!(∇²ϕ)
    return nothing
end

function solve_for_pressure!(pressure, solver::ImmersedPoissonSolver, Δt, U★)
    rhs = solver.rhs
    grid = solver.grid
    arch = architecture(grid)

    rhs_event = launch!(arch, grid, :xyz, calculate_pressure_source_term!,
                        rhs, grid, Δt, U★, dependencies = device_event(arch))

    wait(device(arch), rhs_event)

    # Solve pressure Pressure equation for pressure, given rhs
    solve!(pressure, solver.pcg_solver, rhs)

    return pressure
end
