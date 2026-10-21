using CairoMakie, Serialization
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
main, pr = deserialize("recon_main.jls"), deserialize("recon_thin.jls")
xf, xc = (0:33) ./ 16, ((1:33) .- 0.5) ./ 16
yf, yc = (0:16) ./ 16, ((1:16) .- 0.5) ./ 16
coords = (c = (xf, yc), u = (xc, yc), ζ = (xc, yf), ζᵛ = (xc, yf))
names = (c = "tracer c (cell centers), reconstructed to x-faces",
         u = "u (x-faces), reconstructed to cell centers",
         ζ = "vorticity ζ (x- and y-faces), WENO(order=9)",
         ζᵛ = "vorticity ζ (x- and y-faces), WENOVectorInvariant()")

function right_only(r, name)
    d = log10.(max.(copy(r.Δ[name]), 1e-16))
    d[setdiff(axes(d, 1), r.right[name]), :] .= NaN
    return d
end

fig = Figure(size=(1150, 1300))
colormap = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
ax = Axis(fig[1, 1:2]; title="ζ, vortices in both boxes", aspect=DataAspect(), xlabel="x", ylabel="y")
ζmax = maximum(abs, main.ζ)
heatmap!(ax, xf, yf, main.ζ; colormap, colorrange=(-ζmax, ζmax))
poly!(ax, Rect(1, 0, 1/16, 1), color=solid)
Colorbar(fig[1, 3]; colormap, colorrange=(-ζmax, ζmax), label="ζ")

sequential = cgrad([colorant"#f0efec", colorant"#9ec5f4", colorant"#2a78d6", colorant"#0d366b"])
for (row, name) in enumerate((:c, :u, :ζ, :ζᵛ)), (col, (title, r)) in enumerate((("main", main), ("this PR", pr)))
    x, y = coords[name]
    ax = Axis(fig[row + 1, col]; title = "$title: $(names[name])", titlesize=13, aspect=DataAspect(),
              xlabel = row == 4 ? "x" : "", ylabel = col == 1 ? "y" : "", yticklabelsvisible = col == 1)
    heatmap!(ax, x, y, right_only(r, name); colormap=sequential, colorrange=(-12, 1), nan_color=surface)
    poly!(ax, Rect(1, 0, 1/16, 1), color=solid)
    text!(ax, 0.5, 0.5, text="left box\nchanged", align=(:center, :center), color=ink2)
    text!(ax, 1.65, 0.5, text="max $(round(maximum(r.Δ[name][r.right[name], :]), sigdigits=2))",
          align=(:center, :center), color=ink)
end
Colorbar(fig[2:5, 3]; colormap=sequential, colorrange=(-12, 1), label="log₁₀ |change in the reconstructed value|")
Label(fig[0, 1:3], "Reconstruction in x next to a one-cell wall: change in the right box when only the left box changes",
      fontsize=15, tellwidth=false)
save("reconstruction_leak.png", fig, px_per_unit=2)
