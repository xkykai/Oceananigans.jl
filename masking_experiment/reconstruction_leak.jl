using Oceananigans
using Oceananigans.Advection: _biased_interpolate_xᶠᵃᵃ, _biased_interpolate_xᶜᵃᵃ, materialize_advection, LeftBias, RightBias
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Serialization

# Reconstruct a center field (c), an x-face field (u) and the vorticity (ζ, on x- and y-faces) in x with WENO(order=9)
# next to a one-cell immersed wall, and measure how much the right box depends on the left box.
grid = RectilinearGrid(size=(33, 16), x=(0, 2 + 1/16), y=(0, 1), halo=(6, 6), topology=(Bounded, Bounded, Flat))
grid = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y) -> 1 < x < 1 + 1/16))
scheme = materialize_advection(WENO(order=9), grid)

ξ(x) = x < 1 ? x : x - 1 - 1/16
inside(x) = x < 1 || x > 1 + 1/16

function fields(left)
    A(x) = x < 1 ? left : 1
    u, v, c = XFaceField(grid), YFaceField(grid), CenterField(grid)
    set!(u, (x, y) ->  2π * A(x) * inside(x) * sin(π * ξ(x))^2 * sin(π * y) * cos(π * y))
    set!(v, (x, y) -> -2π * A(x) * inside(x) * sin(π * ξ(x)) * cos(π * ξ(x)) * sin(π * y)^2)
    set!(c, (x, y) -> A(x) * inside(x) * cos(2π * x) * sin(π * y))
    ζ = Field(∂x(v) - ∂y(u))
    compute!(ζ)
    fill_halo_regions!((u, v, c, ζ))
    return (; u, c, ζ)
end

reconstruct(f, ψ, i, j) = max(abs(f(i, j, 1, grid, scheme, LeftBias, ψ)), abs(f(i, j, 1, grid, scheme, RightBias, ψ)))

function reconstructions(F)
    Nx, Ny = 33, 16
    return (c = [reconstruct(_biased_interpolate_xᶠᵃᵃ, F.c, i, j) for i in 1:Nx+1, j in 1:Ny],  # center → x-face
            u = [reconstruct(_biased_interpolate_xᶜᵃᵃ, F.u, i, j) for i in 1:Nx, j in 1:Ny],    # x-face → center
            ζ = [reconstruct(_biased_interpolate_xᶜᵃᵃ, F.ζ, i, j) for i in 1:Nx, j in 1:Ny+1])  # x-face → center
end

Rᴬ, Rᴮ = reconstructions(fields(1)), reconstructions(fields(0))
Δ = map((a, b) -> abs.(a .- b), Rᴬ, Rᴮ)
right = (c = 19:34, u = 18:33, ζ = 18:33)
for name in (:c, :u, :ζ)
    println(name, ": max change of the right-box reconstruction = ", maximum(Δ[name][right[name], :]))
end
F = fields(1)
serialize(ARGS[1], (; Δ, ζ = Array(interior(F.ζ, :, :, 1)), right))
