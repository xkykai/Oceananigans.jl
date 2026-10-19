using Oceananigans
using Oceananigans.Advection: _advective_momentum_flux_Wu, _symmetric_interpolate_xᶠᵃᵃ, _biased_interpolate_xᶠᵃᵃ,
                              δy_V_plus_∂t_σ, δx_U, LeftBias
using Oceananigans.Operators: δzᵃᵃᶜ, Vᶠᶜᶜ
using Oceananigans.ImmersedBoundaries: peripheral_node
using Printf

# Vertical advection of a uniform zonal momentum by a hydrostatic flow over a stepped hill must vanish:
# (1/V) [ (δvˢ + δuᴿ) + δz(w̃ · 1) ], the vector invariant form of ∂z(w) + ∇ₕ·uₕ = 0 at u points
steps(x) = 0.1 * round(5 * exp(-(x - 1/2)^2 / (2 * 0.12^2)))

function uniform_momentum_residual(Bottom, momentum_advection; N=64)
    grid = RectilinearGrid(size=(N, 8, N), x=(0, 1), y=(0, 1), z=(0, 1), halo=(7, 7, 7),
                           topology=(Periodic, Periodic, Bounded))
    ibg = ImmersedBoundaryGrid(grid, Bottom((x, y) -> steps(x) + 0.013))
    model = HydrostaticFreeSurfaceModel(ibg; momentum_advection, free_surface=ImplicitFreeSurface())
    set!(model, u = (x, y, z) -> 1 + 0.5 * sin(2π * x) * cos(π * z))

    scheme = model.advection.momentum
    u, v, w = model.velocities
    uniform = XFaceField(ibg)
    parent(uniform) .= 1

    residual(i, k) = (_symmetric_interpolate_xᶠᵃᵃ(i, 1, k, ibg, scheme, scheme.upwinding.cross_scheme, δy_V_plus_∂t_σ, u, v) +
                      _biased_interpolate_xᶠᵃᵃ(i, 1, k, ibg, scheme, scheme.divergence_scheme, LeftBias, δx_U, scheme.upwinding.δU_stencil, u, v) +
                      δzᵃᵃᶜ(i, 1, k, ibg, _advective_momentum_flux_Wu, scheme.vertical_advection_scheme, w, uniform)) / Vᶠᶜᶜ(i, 1, k, ibg)

    R = [peripheral_node(i, 1, k, ibg, Face(), Center(), Center()) ? 0.0 : residual(i, k) for i in 1:N, k in 1:N]
    scale = maximum(abs, interior(u)) * N
    return maximum(abs, R) / scale, count(r -> abs(r) > 1e-10 * scale, R)
end

for (Bottom, bname) in ((GridFittedBottom, "full"), (PartialCellBottom, "partial")),
    (name, advection) in (("WENOVectorInvariant()", WENOVectorInvariant()),
                          ("VectorInvariant, UpwindBiased(3)", VectorInvariant(vertical_advection_scheme=UpwindBiased(order=3),
                                                                                divergence_scheme=UpwindBiased(order=3))))
    R, n = uniform_momentum_residual(Bottom, advection)
    @printf("%-8s %-33s max residual / (max|u|/Δx) = %.2e  on %d u-cells\n", bname, name, R, n)
end
