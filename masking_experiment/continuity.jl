using Oceananigans
using Oceananigans.Advection: div_𝐯u, div_𝐯w,
                              _advective_momentum_flux_Uu, _advective_momentum_flux_Wu,
                              _advective_momentum_flux_Uw, _advective_momentum_flux_Ww
using Oceananigans.ImmersedBoundaries: peripheral_node, immersed_peripheral_node, inactive_node
using Oceananigans.Operators: Ax_qᶠᶜᶜ, Az_qᶜᶜᶠ, Vᶠᶜᶜ, Vᶜᶜᶠ, divᶜᶜᶜ
using Oceananigans.Solvers: ConjugateGradientPoissonSolver
using Serialization

outdir = ARGS[1]
mkpath(outdir)

staircase(x) = (5 + sum(tanh(40 * (x - n / 6)) for n in 1:5)) / 20

const c = Center()
const f = Face()

function projected_flow(Bottom, scheme; N=64)
    grid = RectilinearGrid(size=(N, N), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))
    ibg = ImmersedBoundaryGrid(grid, Bottom(staircase))
    pressure_solver = ConjugateGradientPoissonSolver(ibg; reltol=1e-13, abstol=0, maxiter=2000)
    model = NonhydrostaticModel(ibg; advection=scheme, pressure_solver)
    set!(model, u = (x, z) -> 1 + cos(3x) * sin(2π * z), w = (x, z) -> sin(4π * x) * cos(π * z))
    return model
end

# Transport through a momentum-cell face that the momentum flux uses (advected quantity ≡ 1),
# and the volume that crosses the open part of that face on the tracer grid.
function face_audit(model, ones_field)
    ibg = model.grid
    scheme = model.advection.momentum
    u, v, w = model.velocities
    Nx, Nz = size(ibg, 1), size(ibg, 3)
    open(i, k, ℓx, ℓz) = !peripheral_node(i, 1, k, ibg, ℓx, c, ℓz)

    rows = []
    for i in 1:Nx+1, k in 1:Nz+1
        # Wu and Uw at fcf
        if i ≤ Nx+1 && k ≤ Nz+1 && immersed_peripheral_node(i, 1, k, ibg, f, c, f)
            Tmom = _advective_momentum_flux_Wu(i, 1, k, ibg, scheme, w, ones_field)
            Topen = (Az_qᶜᶜᶠ(i-1, 1, k, ibg, w) + Az_qᶜᶜᶠ(i, 1, k, ibg, w)) / 2
            nopen = open(i-1, k, c, f) + open(i, k, c, f)
            push!(rows, (:Wu, i, k, Tmom, Topen, nopen))

            Tmom = _advective_momentum_flux_Uw(i, 1, k, ibg, scheme, u, ones_field)
            Topen = (Ax_qᶠᶜᶜ(i, 1, k-1, ibg, u) + Ax_qᶠᶜᶜ(i, 1, k, ibg, u)) / 2
            nopen = open(i, k-1, f, c) + open(i, k, f, c)
            push!(rows, (:Uw, i, k, Tmom, Topen, nopen))
        end
        # Uu and Ww at ccc
        if i ≤ Nx && k ≤ Nz && immersed_peripheral_node(i, 1, k, ibg, c, c, c)
            Tmom = _advective_momentum_flux_Uu(i, 1, k, ibg, scheme, u, ones_field)
            Topen = (Ax_qᶠᶜᶜ(i, 1, k, ibg, u) + Ax_qᶠᶜᶜ(i+1, 1, k, ibg, u)) / 2
            nopen = open(i, k, f, c) + open(i+1, k, f, c)
            push!(rows, (:Uu, i, k, Tmom, Topen, nopen))

            Tmom = _advective_momentum_flux_Ww(i, 1, k, ibg, scheme, w, ones_field)
            Topen = (Az_qᶜᶜᶠ(i, 1, k, ibg, w) + Az_qᶜᶜᶠ(i, 1, k+1, ibg, w)) / 2
            nopen = open(i, k, c, f) + open(i, k+1, c, f)
            push!(rows, (:Ww, i, k, Tmom, Topen, nopen))
        end
    end
    return rows
end

function diagnose(Bottom, scheme)
    model = projected_flow(Bottom, scheme)
    ibg = model.grid
    u, v, w = model.velocities
    U = model.velocities
    Nx, Nz = size(ibg, 1), size(ibg, 3)

    ones_field = Field{Face, Center, Center}(ibg)
    parent(ones_field) .= 1

    # Net advecting volume flux per unit volume of each prognostic momentum cell
    Du = fill(NaN, Nx+1, Nz)
    Dw = fill(NaN, Nx, Nz+1)
    for i in 1:Nx+1, k in 1:Nz
        peripheral_node(i, 1, k, ibg, f, c, c) || (Du[i, k] = div_𝐯u(i, 1, k, ibg, model.advection.momentum, U, ones_field))
    end
    for i in 1:Nx, k in 1:Nz+1
        peripheral_node(i, 1, k, ibg, c, c, f) || (Dw[i, k] = div_𝐯w(i, 1, k, ibg, model.advection.momentum, U, ones_field))
    end

    Dc = [inactive_node(i, 1, k, ibg, c, c, c) ? NaN : divᶜᶜᶜ(i, 1, k, ibg, u, v, w) for i in 1:Nx, k in 1:Nz]
    inside = maximum(abs(u[i, 1, k]) for i in 1:Nx+1, k in 1:Nz if inactive_node(i, 1, k, ibg, f, c, c); init=0.0)
    inside = max(inside, maximum(abs(w[i, 1, k]) for i in 1:Nx, k in 1:Nz+1 if inactive_node(i, 1, k, ibg, c, c, f); init=0.0))

    scale = maximum(abs, interior(u)) / minimum_xspacing(ibg)

    return (; Du, Dw, Dc, scale, inside, audit = face_audit(model, ones_field),
              u = Array(interior(u, :, 1, :)), w = Array(interior(w, :, 1, :)),
              bottom = [staircase(x) for x in xnodes(ibg, c)],
              x = collect(xnodes(ibg, c)), z = collect(znodes(ibg, c)),
              xf = collect(xnodes(ibg, f)), zf = collect(znodes(ibg, f)),
              Δz = minimum_zspacing(ibg))
end

using Oceananigans.Advection: ExplicitTimeDiscretization
weno_with_second_order_advecting_velocity() =
    Oceananigans.Advection.WENO{3, Float64, Nothing}(nothing, WENO(order=3), Centered(order=2), ExplicitTimeDiscretization())

results = Dict()
for (bname, Bottom) in (("full", GridFittedBottom), ("partial", PartialCellBottom)),
    (sname, scheme) in (("Centered", Centered()), ("WENO", WENO()), ("WENO-C2", weno_with_second_order_advecting_velocity()))

    r = diagnose(Bottom, scheme)
    results[(bname, sname)] = r
    max_D = max(maximum(abs, filter(!isnan, r.Du)), maximum(abs, filter(!isnan, r.Dw))) / r.scale
    max_div = maximum(abs, filter(!isnan, r.Dc)) / r.scale
    nonzero = [a for a in r.audit if a[4] != 0]
    closed  = [a for a in r.audit if a[6] == 0]
    mismatch = maximum(abs(a[4] - a[5]) for a in r.audit; init=0.0) / maximum(abs(a[5]) for a in r.audit)
    carrying = count(a -> abs(a[5]) > 1e-12, r.audit)
    println(rpad("$bname $sname", 18),
            " max|D|/scale = ", round(max_D, sigdigits=3),
            "  max|∇·u|/scale = ", round(max_div, sigdigits=3),
            "  |u| inside solid = ", r.inside,
            "  peripheral faces = ", length(r.audit),
            "  with flow through open part = ", carrying, "  with momentum flux ≠ 0 = ", length(nonzero),
            "  fully closed = ", length(closed),
            "  max|T| on closed = ", maximum(abs(a[4]) for a in closed; init=0.0),
            "  max|Tmom - Topen|/max|Topen| = ", round(mismatch, sigdigits=3))
end

serialize(joinpath(outdir, "continuity.jls"), results)
