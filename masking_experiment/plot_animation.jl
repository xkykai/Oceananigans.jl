using CairoMakie
using Serialization

masked = deserialize("anim/masked_2d.jls")
pr     = deserialize("anim/pr_2d.jls")

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
blue, orange = colorant"#2a78d6", colorant"#eb6834"

set_theme!(fontsize=15, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false,
                 leftspinecolor=ink2, bottomspinecolor=ink2, topspinevisible=false, rightspinevisible=false))

speed(frame) = sqrt.(((frame.u[1:end-1, :] .+ frame.u[2:end, :]) ./ 2) .^ 2 .+
                     ((frame.w[:, 1:end-1] .+ frame.w[:, 2:end]) ./ 2) .^ 2)

n = Observable(1)
nframes = min(length(masked.frames), length(pr.frames))

fig = Figure(size=(1200, 760))
colormap = cgrad([colorant"#fcfcfb", colorant"#9ec5f4", colorant"#2a78d6", colorant"#104281", colorant"#0b0b0b"])
for (col, (title, run)) in enumerate((("PR + masking", masked), ("PR", pr)))
    ax = Axis(fig[1, col]; title, aspect=DataAspect(), limits=((0, 1), (0, 1)),
              xlabel="x", ylabel = col == 1 ? "z" : "", yticklabelsvisible = col == 1)
    heatmap!(ax, run.x, run.z, @lift(speed(run.frames[$n])); colormap, colorrange=(0, 0.4), highclip=colorant"#e34948")
    contour!(ax, run.x, run.z, @lift(run.frames[$n].b); levels=[-0.5, -0.1], color=ink2, linewidth=1)
    band!(ax, run.x, zero(run.bottom), run.bottom, color=solid)
end
Colorbar(fig[1, 3], limits=(0, 0.4), colormap=colormap, highclip=colorant"#e34948", label="speed")

ax = Axis(fig[2, 1:2]; xlabel="t", ylabel="max |u|", limits=((0, 6), (0, 1.8)), height=150)
lines!(ax, masked.t, masked.U, color=orange, linewidth=2, label="PR + masking")
lines!(ax, pr.t, pr.U, color=blue, linewidth=2, label="PR")
vlines!(ax, @lift(pr.frames[$n].t), color=ink2, linewidth=1)
axislegend(ax, position=:lt, framevisible=false)

Label(fig[0, 1:2], @lift("Cold blob over a partial-cell staircase, WENO(), no viscosity, t = $(round(pr.frames[$n].t, digits=2))"),
      fontsize=17, tellwidth=false)

record(fig, "staircase_convection_masking.mp4", 1:nframes; framerate=20) do i
    n[] = i
end
n[] = findfirst(f -> f.t ≥ 3, pr.frames)
save("staircase_convection_masking_t3.png", fig, px_per_unit=2)
