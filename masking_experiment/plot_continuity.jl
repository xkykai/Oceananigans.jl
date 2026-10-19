using CairoMakie
using Serialization

masked = deserialize("cont_masked/continuity.jls")
pr     = deserialize("cont_pr/continuity.jls")

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
blue, orange, aqua = colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a"

set_theme!(fontsize=15, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false,
                 leftspinecolor=ink2, bottomspinecolor=ink2, topspinevisible=false, rightspinevisible=false,
                 xtickcolor=ink2, ytickcolor=ink2))

function draw_bottom!(ax, d, bname)
    if bname == "full"
        h = [d.zf[findfirst(>(b), d.z)] for b in d.bottom]
    else
        h = d.bottom
    end
    xs = vcat([[d.xf[i], d.xf[i+1]] for i in eachindex(h)]...)
    hs = vcat([[h[i], h[i]] for i in eachindex(h)]...)
    band!(ax, xs, zero(hs), hs, color=solid)
    lines!(ax, xs, hs, color=ink2, linewidth=1)
end

function nodes_with_residual(d)
    pts, vals = Point2f[], Float64[]
    for I in CartesianIndices(d.Du)
        v = d.Du[I] / d.scale
        (isnan(v) || abs(v) < 1e-10) && continue
        push!(pts, Point2f(d.xf[I[1]], d.z[I[2]])); push!(vals, abs(v))
    end
    for I in CartesianIndices(d.Dw)
        v = d.Dw[I] / d.scale
        (isnan(v) || abs(v) < 1e-10) && continue
        push!(pts, Point2f(d.x[I[1]], d.zf[I[2]])); push!(vals, abs(v))
    end
    return pts, vals
end

#####
##### Figure 1: momentum-cell continuity residual
#####

cases = [("PR + masking", masked, "WENO"),
         ("PR", pr, "WENO"),
         ("PR, WENO with Centered(order=2)\nadvecting velocity", pr, "WENO-C2")]

colormap = cgrad([colorant"#cde2fb", colorant"#5598e7", colorant"#184f95", colorant"#0d366b"])
colorrange = (-4, 0)

fig = Figure(size=(1250, 720))
for (row, (bname, label)) in enumerate((("full", "GridFittedBottom"), ("partial", "PartialCellBottom")))
    for (col, (title, results, scheme)) in enumerate(cases)
        d = results[(bname, scheme)]
        ax = Axis(fig[row, col]; limits=((0.05, 0.62), (0.0, 0.42)), aspect=DataAspect(),
                  title = row == 1 ? title : "", titlegap=10,
                  xlabel = row == 2 ? "x" : "", ylabel = col == 1 ? "$label\nz" : "",
                  xticklabelsvisible = row == 2, yticklabelsvisible = col == 1)
        draw_bottom!(ax, d, bname)
        pts, vals = nodes_with_residual(d)
        if isempty(vals)
            text!(ax, 0.335, 0.33, text="|D| < 10⁻¹² everywhere", align=(:center, :center), color=ink2)
        else
            scatter!(ax, pts, color=log10.(vals); colormap, colorrange, markersize=9,
                     strokecolor=surface, strokewidth=1)
            text!(ax, 0.07, 0.40, text="max |D| = $(round(maximum(vals), sigdigits=2))", align=(:left, :top), color=ink2)
        end
    end
end
Colorbar(fig[1:2, 4]; colormap, colorrange, label="log₁₀ |D| / (max|u| / Δx)", ticks=-4:0)
Label(fig[0, 1:3], "Net advecting volume flux D of each momentum cell (must vanish for a divergence-free flow)",
      fontsize=17, tellwidth=false)
save("momentum_cell_continuity.png", fig, px_per_unit=2)

#####
##### Figure 2: transport the momentum flux uses on immersed-peripheral faces
#####

fig = Figure(size=(1200, 820))
series = [("PR + masking", masked, "WENO", orange, :rect),
          ("PR, Centered", pr, "Centered", blue, :circle),
          ("PR, WENO", pr, "WENO", aqua, :utriangle)]
for (row, (bname, label)) in enumerate((("full", "GridFittedBottom"), ("partial", "PartialCellBottom")))
    scale = maximum(abs(a[5]) for a in pr[(bname, "Centered")].audit)
    for (col, (name, results, scheme, color, marker)) in enumerate(series)
        d = results[(bname, scheme)]
        ax = Axis(fig[row, col]; aspect=1, limits=((-1.1, 1.1), (-1.1, 1.1)),
                  title = row == 1 ? name : "",
                  xlabel = row == 2 ? "volume flux through open part of face" : "",
                  ylabel = col == 1 ? "$label\ntransport used by momentum flux" : "",
                  xticklabelsvisible = row == 2, yticklabelsvisible = col == 1)
        lines!(ax, [-1.1, 1.1], [-1.1, 1.1], color=ink2, linestyle=:dash, linewidth=1)
        rows = [a for a in d.audit if a[4] != 0 || abs(a[5]) > 1e-12]
        scatter!(ax, [a[5] / scale for a in rows], [a[4] / scale for a in rows];
                 color, marker, markersize=10, strokecolor=surface, strokewidth=1)
        closed = [a for a in d.audit if a[6] == 0 && a[4] != 0]
        if !isempty(closed)
            scatter!(ax, zeros(length(closed)), [a[4] / scale for a in closed];
                     color=:transparent, markersize=22, strokecolor=ink, strokewidth=1.5)
            text!(ax, 0.08, -0.35, text="$(length(closed)) faces with no open area\ncarry transport", 
                  align=(:left, :top), color=ink, fontsize=13)
        end
        text!(ax, -1.0, 1.0, text="$(length(rows)) faces", align=(:left, :top), color=ink2, fontsize=13)
    end
end
Label(fig[0, 1:3], "Immersed-peripheral momentum faces (the faces the masking zeroes), normalized by the largest open-face flux",
      fontsize=17, tellwidth=false)
save("immersed_face_transport.png", fig, px_per_unit=2)
