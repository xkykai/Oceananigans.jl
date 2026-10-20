using CairoMakie
using Serialization

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
blue, orange = colorant"#2a78d6", colorant"#eb6834"

set_theme!(fontsize=15, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false,
                 leftspinecolor=ink2, bottomspinecolor=ink2, topspinevisible=false, rightspinevisible=false,
                 xtickcolor=ink2, ytickcolor=ink2))

main = deserialize("cont_base/continuity.jls")
pr   = deserialize("cont_pr/continuity.jls")

#####
##### Face audit and momentum-cell continuity (WENO, partial cells)
#####

function draw_bottom!(ax, d)
    xs = vcat([[d.xf[i], d.xf[i+1]] for i in eachindex(d.bottom)]...)
    hs = vcat([[h, h] for h in d.bottom]...)
    band!(ax, xs, zero(hs), hs, color=solid)
    lines!(ax, xs, hs, color=ink2, linewidth=1)
end

function nodes_with_residual(d)
    pts, vals = Point2f[], Float64[]
    for I in CartesianIndices(d.Du)
        v = d.Du[I] / d.scale
        (isnan(v) || abs(v) < 1e-10) || (push!(pts, Point2f(d.xf[I[1]], d.z[I[2]])); push!(vals, abs(v)))
    end
    for I in CartesianIndices(d.Dw)
        v = d.Dw[I] / d.scale
        (isnan(v) || abs(v) < 1e-10) || (push!(pts, Point2f(d.x[I[1]], d.zf[I[2]])); push!(vals, abs(v)))
    end
    return pts, vals
end

fig = Figure(size=(1250, 880))
colormap = cgrad([colorant"#cde2fb", colorant"#5598e7", colorant"#184f95", colorant"#0d366b"])
for (col, (title, results, color, marker)) in enumerate((("main", main, orange, :rect), ("this PR", pr, blue, :circle)))
    d = results[("partial", "WENO")]
    scale = maximum(abs(a[5]) for a in pr[("partial", "Centered")].audit)
    ax = Axis(fig[1, col]; title, aspect=1, limits=((-1.1, 1.1), (-1.1, 1.1)),
              xlabel="volume flux through the open part of the face",
              ylabel = col == 1 ? "transport used by the momentum flux" : "", yticklabelsvisible = col == 1)
    lines!(ax, [-1.1, 1.1], [-1.1, 1.1], color=ink2, linestyle=:dash, linewidth=1)
    rows = [a for a in d.audit if a[4] != 0 || abs(a[5]) > 1e-12]
    scatter!(ax, [a[5] / scale for a in rows], [a[4] / scale for a in rows]; color, marker, markersize=10,
             strokecolor=surface, strokewidth=1)

    ax = Axis(fig[2, col]; aspect=DataAspect(), limits=((0.05, 0.62), (0.0, 0.42)), xlabel="x",
              ylabel = col == 1 ? "z" : "", yticklabelsvisible = col == 1)
    draw_bottom!(ax, d)
    pts, vals = nodes_with_residual(d)
    scatter!(ax, pts, color=log10.(vals); colormap, colorrange=(-4, 0), markersize=9, strokecolor=surface, strokewidth=1)
    text!(ax, 0.07, 0.40, text="max = $(round(maximum(vals), sigdigits=2))", align=(:left, :top), color=ink2)
end
Colorbar(fig[2, 3]; colormap, colorrange=(-4, 0), label="log₁₀ net volume flux of momentum cell\n/ (max|u| / Δx)")
Label(fig[0, 1:2], "Faces touching the immersed boundary (top) and momentum-cell continuity (bottom), WENO(), PartialCellBottom",
      fontsize=16, tellwidth=false)
save("pr6187_continuity.png", fig, px_per_unit=2)

#####
##### Convergence over a stepped sill: flux form vs vector invariant
#####

Ns = (64, 128, 256, 512)
fig = Figure(size=(1250, 640))
for (row, (file, scheme)) in enumerate((("stepsweno_errors.jls", "WENO()"), ("steps_errors.jls", "WENOVectorInvariant()")))
    results = deserialize(file)
    for (col, (bottom, name)) in enumerate((("full", :u), ("full", :b), ("partial", :u), ("partial", :b)))
        label = name == :u ? "u − U" : "b − N²z"
        ax = Axis(fig[row, col]; xscale=log2, yscale=log10, xticks=collect(Ns),
                  title = row == 1 ? "$(bottom == "full" ? "GridFittedBottom" : "PartialCellBottom"), $label" : "",
                  xlabel = row == 2 ? "N" : "", ylabel = col == 1 ? "$scheme\nrelative error, t = 4" : "")
        for (build, color, marker, legend) in (("base", orange, :rect, "main"), ("pr", blue, :circle, "this PR"))
            for (reference, style) in (("pr", :solid), ("base", :dash))
                e = [getproperty(results[(bottom, reference, 4.0, N, build)], name) for N in Ns]
                scatterlines!(ax, collect(Ns), e; color, marker, markersize=9, linewidth=2, linestyle=style,
                              label = reference == "pr" ? legend : nothing)
            end
        end
        row == 1 && col == 1 && axislegend(ax, position=:lb, framevisible=false)
    end
end
Label(fig[0, 1:4], "Stratified flow over a stepped sill, error against 1024² of the same geometry (solid: PR reference, dashed: main reference)",
      fontsize=16, tellwidth=false)
save("pr6187_convergence.png", fig, px_per_unit=2)
