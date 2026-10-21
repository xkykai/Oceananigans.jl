using CairoMakie, Serialization
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
main, pr = deserialize("recon_main.jls"), deserialize("recon_thin.jls")
xf, xc = (0:33) ./ 16, ((1:33) .- 0.5) ./ 16
yf, yc = (0:16) ./ 16, ((1:16) .- 0.5) ./ 16
fieldnames = (:c, :u, :ζ, :ζᵛ)
coords = (c = (xf, yc), u = (xc, yc), ζ = (xc, yf), ζᵛ = (xc, yf))
titles = (c = "tracer c (centers)", u = "u (x-faces)", ζ = "ζ (x- and y-faces)\nWENO(order=9)", ζᵛ = "ζ (x- and y-faces)\nWENOVectorInvariant()")

function right_only(r, name)
    d = log10.(max.(copy(r.Δ[name]), 1e-16))
    d[setdiff(axes(d, 1), r.right[name]), :] .= NaN
    return d
end

fig = Figure(size=(1500, 820))
colormap = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
ζmax = maximum(abs, main.ζ)
ax = Axis(fig[1, 2:3]; title="ζ, vortices in both boxes", aspect=DataAspect(), xlabel="x", ylabel="y")
heatmap!(ax, xf, yf, main.ζ; colormap, colorrange=(-ζmax, ζmax))
poly!(ax, Rect(1, 0, 1/16, 1), color=solid)
Colorbar(fig[1, 4]; colormap, colorrange=(-ζmax, ζmax), label="ζ", tellwidth=false, halign=:left, width=15)

sequential = cgrad([colorant"#f0efec", colorant"#9ec5f4", colorant"#2a78d6", colorant"#0d366b"])
for (row, (label, r)) in enumerate((("main", main), ("this PR", pr))), (col, name) in enumerate(fieldnames)
    x, y = coords[name]
    ax = Axis(fig[row + 1, col]; title = row == 1 ? titles[name] : "", aspect=DataAspect(),
              xlabel = row == 2 ? "x" : "", ylabel = col == 1 ? "$label\ny" : "",
              yticklabelsvisible = col == 1, xticklabelsvisible = row == 2)
    heatmap!(ax, x, y, right_only(r, name); colormap=sequential, colorrange=(-12, 1), nan_color=surface)
    poly!(ax, Rect(1, 0, 1/16, 1), color=solid)
    text!(ax, 0.5, 0.5, text="left box\nchanged", align=(:center, :center), color=ink2, fontsize=12)
    text!(ax, 1.65, 0.5, text="max $(round(maximum(r.Δ[name][r.right[name], :]), sigdigits=2))",
          align=(:center, :center), color=ink, fontsize=13)
end
Colorbar(fig[2:3, 5]; colormap=sequential, colorrange=(-12, 1), label="log₁₀ |change in the reconstructed value|")
Label(fig[0, 1:5], "Reconstruction in x next to a one-cell wall: change in the right box when only the left box changes",
      fontsize=16, tellwidth=false)
rowsize!(fig.layout, 1, Relative(0.28))
rowgap!(fig.layout, 10)
resize_to_layout!(fig)
save("reconstruction_leak.png", fig, px_per_unit=2)
