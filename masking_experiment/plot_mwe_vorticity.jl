using CairoMakie, Serialization
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
xf = ((1:33) .- 0.5) ./ 16
zc = (0:16) ./ 16
main, pr = deserialize("mwe_vort_main.jls"), deserialize("mwe_vort_thin.jls")

right_box_difference(r) = (d = abs.(r.vᴬ .- r.vᴮ); d[1:17, :] .= NaN; log10.(max.(d, 1e-16)))

fig = Figure(size=(1250, 360))
colormap = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
ax = Axis(fig[1, 1]; title="v, vortices in both boxes", xlabel="x", ylabel="y", aspect=DataAspect())
heatmap!(ax, xf, zc, pr.vᴬ; colormap, colorrange=(-3, 3))
poly!(ax, Rect(1, 0, 1/16, 1), color=solid)
Colorbar(fig[1, 2]; colormap, colorrange=(-3, 3), label="v")

sequential = cgrad([colorant"#f0efec", colorant"#9ec5f4", colorant"#2a78d6", colorant"#0d366b"])
for (col, (title, r)) in enumerate((("main", main), ("this PR", pr)))
    ax = Axis(fig[1, col + 2]; title="$title: |Δv| in the right box", xlabel="x", aspect=DataAspect(),
              yticklabelsvisible=false)
    heatmap!(ax, xf, zc, right_box_difference(r); colormap=sequential, colorrange=(-12, -1), nan_color=surface)
    poly!(ax, Rect(1, 0, 1/16, 1), color=solid)
    text!(ax, 0.5, 0.5, text="left vortex\non / off", align=(:center, :center), color=ink2)
end
Colorbar(fig[1, 5]; colormap=sequential, colorrange=(-12, -1), label="log₁₀ |Δv|")
Label(fig[0, 1:5], "One-cell wall, VectorInvariant(vorticity_scheme=WENO(order=9)): change in the right box when the left vortex is removed (t = 1)", fontsize=16, tellwidth=false)
save("thin_wall_vorticity_leak.png", fig, px_per_unit=2)
