using CairoMakie, Serialization

E = ARGS[1]
path(v, name) = joinpath(E, "out_$v", name * ".jls")
load(v, name) = deserialize(path(v, name))
labels = ("base" => "main (masked)", "pr" => "this PR (unmasked)")
colors = Dict("base" => :firebrick, "pr" => :royalblue)

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20
centers(N) = ((1:N) .- 1/2) ./ N

function center_u(r, h)
    uc = (r.u[1:end-1, :] .+ r.u[2:end, :]) ./ 2
    x, z = centers(size(uc, 1)), centers(size(uc, 2))
    uc[[zj < h(xi) for xi in x, zj in z]] .= NaN
    return x, z, uc
end

set_theme!(Theme(fontsize=18))

# Kinetic energy conservation
fig = Figure(size=(1300, 1000))
ax1 = Axis(fig[1, 1]; ylabel="KE(t) / KE(0) − 1",
           title="Inviscid, unforced flow over a GridFittedBottom staircase, Centered() advection")
ax2 = Axis(fig[2, 1]; xlabel="t", ylabel="|KE(t) / KE(0) − 1|", yscale=log10)
for (v, label) in labels
    for (name, style, width, suffix) in (("ke_staircase", :solid, 3, "staircase"),
                                         ("ke_staircase_half_dt", :dot, 3, "staircase, Δt / 2"),
                                         ("ke_flat", :dash, 2, "flat bottom"))
        r = load(v, name)
        drift = r.E ./ r.E[1] .- 1
        lines!(ax1, r.t, drift; color=colors[v], linestyle=style, linewidth=width, label="$label, $suffix")
        keep = 2:length(r.t)
        lines!(ax2, r.t[keep], max.(abs.(drift[keep]), 1e-9); color=colors[v], linestyle=style, linewidth=width)
    end
end
axislegend(ax1; position=:lt)
ylims!(ax2, 1e-8, 1e-1)
save(joinpath(E, "kinetic_energy.png"), fig)

# Convection over a partial-cell staircase
convection = isfile(path("pr", "convection_low_blob")) ? "convection_low_blob" : "convection"
c = Dict(v => load(v, convection) for v in ("base", "pr"))
ntimes = minimum(length(c[v].snapshots) for v in keys(c))
fig = Figure(size=(380 * ntimes + 120, 1150))
for (row, (v, label)) in enumerate(labels), n in 1:ntimes
    snap = c[v].snapshots[n]
    x, z, u = center_u(snap, staircase)
    a = Axis(fig[row, n]; title="$label, t = $(round(snap.t, digits=2))", aspect=1,
             xlabel = row == 2 ? "x" : "", ylabel = n == 1 ? "z" : "")
    hm = heatmap!(a, x, z, u; colormap=:balance, colorrange=(-0.5, 0.5), nan_color=:gray30)
    contour!(a, x, z, snap.b[:, :]; levels=[-0.5, -0.1], color=:black, linewidth=1)
    (row, n) == (1, ntimes) && Colorbar(fig[1:2, ntimes + 1], hm; label="u")
end
ax = Axis(fig[3, 1:ntimes]; xlabel="t", ylabel="max |u|", yscale=log10,
          title="Cold blob sinking onto a PartialCellBottom staircase, WENO(order=5)")
for (v, label) in labels
    r = c[v]; g = load(v, "convection_test_grid")
    lines!(ax, r.t[2:end], r.U[2:end]; color=colors[v], linewidth=3, label="$label, 128 × 128")
    lines!(ax, g.t[2:end], g.U[2:end]; color=colors[v], linewidth=2, linestyle=:dash, label="$label, 16 × 128")
    isfile(path(v, "convection_3d")) || continue
    d = load(v, "convection_3d")
    lines!(ax, d.t[2:end], d.U[2:end]; color=colors[v], linewidth=2, linestyle=:dot, label="$label, 16 × 16 × 128 (3D)")
end
ylims!(ax, 0.05, 1e3)
axislegend(ax; position=:rt)
save(joinpath(E, "staircase_convection.png"), fig)
