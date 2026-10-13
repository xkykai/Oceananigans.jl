using CairoMakie, Serialization

E = ARGS[1]
load(path) = deserialize(joinpath(E, path * ".jls"))
colors = Dict("main" => :firebrick, "this PR" => :royalblue)

staircase(ξ) = (5 + sum(tanh(40 * (ξ - n / 6)) for n in 1:5)) / 20
centers(N) = ((1:N) .- 1/2) ./ N

function center_u(snapshot)
    u = (snapshot.u[1:end-1, :] .+ snapshot.u[2:end, :]) ./ 2
    x, z = centers(size(u, 1)), centers(size(u, 2))
    u[[zj < staircase(xi) for xi in x, zj in z]] .= NaN
    return x, z, u
end

rows = (("main", "partial cells", "conv_base/conv_128_partial"),
        ("this PR", "partial cells", "conv_pr/conv_128_partial"),
        ("main", "full cells", "conv_base/conv_128_full"),
        ("this PR", "full cells", "conv_pr/conv_128_full"))

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1300, 2050))

for (row, (label, cells, path)) in enumerate(rows)
    snapshots = unique(s -> round(Int, s.t), load(path).snapshots)
    for (n, snapshot) in enumerate(snapshots)
        x, z, u = center_u(snapshot)
        ax = Axis(fig[row, n]; title="$label, $cells, t = $(round(Int, snapshot.t))", aspect=1,
                  xlabel = row == 4 ? "x" : "", ylabel = n == 1 ? "z" : "")
        heatmap!(ax, x, z, u; colormap=:balance, colorrange=(-0.5, 0.5), nan_color=:gray30)
        contour!(ax, x, z, snapshot.b; levels=[-0.5, -0.1], color=:black, linewidth=1)
    end
end
Colorbar(fig[1:4, 4]; colormap=:balance, colorrange=(-0.5, 0.5), label="u")

series = (("16 × 128", ("out_base/convection_test_grid", "out_base/convection_test_grid_full"), ("conv_pr/conv_16_partial", "conv_pr/conv_16_full")),
          ("128 × 128", ("conv_base/conv_128_partial", "conv_base/conv_128_full"), ("conv_pr/conv_128_partial", "conv_pr/conv_128_full")),
          ("16 × 16 × 128 (#6186)", ("out_base/convection_3d", "out_base/convection_3d_full"), ("conv_pr/conv_3d_partial", "conv_pr/conv_3d_full")))

g = fig[5, 1:4] = GridLayout()
for (n, (title, main_paths, pr_paths)) in enumerate(series)
    ax = Axis(g[1, n]; title, xlabel="t", ylabel = n == 1 ? "max |u|" : "", yscale=log10)
    for (label, paths) in (("main", main_paths), ("this PR", pr_paths)), (path, style, cells) in zip(paths, (:solid, :dash), ("partial cells", "full cells"))
        r = load(path)
        lines!(ax, r.t[2:end], r.U[2:end]; color=colors[label], linestyle=style, linewidth=2.5, label="$label, $cells")
    end
    ylims!(ax, 0.05, 1e3)
    n == 3 && axislegend(ax; position=:lt)
end
rowsize!(fig.layout, 5, Relative(0.22))

save(joinpath(E, "staircase_convection.png"), fig)
