using CairoMakie

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
blue, red, aqua = colorant"#2a78d6", colorant"#e34948", colorant"#1baf7a"

set_theme!(fontsize=15, textcolor=ink, backgroundcolor=surface)

# One row: cells 0..8 along the stencil direction, solid cells shaded, data points, checked nodes
function draw_row!(ax, y, solid_cells; data, checked, label, verdict, verdict_color)
    for c in solid_cells
        poly!(ax, Rect(c, y - 0.35, 1, 0.7), color=solid, strokecolor=ink2, strokewidth=1)
    end
    for x in 0:8
        lines!(ax, [x, x], [y - 0.35, y + 0.35], color=ink2, linewidth=1)
    end
    lines!(ax, [0, 8], [y - 0.35, y - 0.35], color=ink2, linewidth=1)
    lines!(ax, [0, 8], [y + 0.35, y + 0.35], color=ink2, linewidth=1)

    flagged = [x for x in checked if any(c -> c < x < c + 1, solid_cells) || (isinteger(x) && (x - 1) in solid_cells && x in solid_cells)]
    scatter!(ax, data, fill(y, length(data)), color=blue, markersize=13)
    scatter!(ax, checked, fill(y - 0.62, length(checked)), marker=:rect, color=:transparent,
             strokecolor=ink, strokewidth=1.5, markersize=13)
    isempty(flagged) || scatter!(ax, flagged, fill(y - 0.62, length(flagged)), marker=:xcross, color=red, markersize=15)

    text!(ax, -0.25, y, text=label, align=(:right, :center), fontsize=14)
    text!(ax, 8.25, y, text=verdict, align=(:left, :center), color=verdict_color, fontsize=14)
end

fig = Figure(size=(1250, 560))
ax = Axis(fig[1, 1]; limits=((-5.2, 12.2), (-7.0, 0.9)), aspect=DataAspect(), backgroundcolor=surface)
hidedecorations!(ax); hidespines!(ax)

faces  = 2:6        # right-biased 5-point stencil of face values for the cell centred at 3.5
cells  = 2.5:1:5.5  # cells between those faces
centers = 1.5:1:5.5 # 5-point stencil of center values

draw_row!(ax, 0, [4]; data=faces, checked=faces,
          label="face values (u in x, ζ)\nmain: check faces", verdict="no face inactive:\nstencil crosses the wall", verdict_color=red)
draw_row!(ax, -2, [4]; data=faces, checked=cells,
          label="face values (u in x, ζ)\nthis PR: check cells between faces", verdict="solid cell found:\norder reduced", verdict_color=aqua)
draw_row!(ax, -4, [4, 5]; data=faces, checked=faces,
          label="face values, two-cell wall\nmain: check faces", verdict="shared face inactive:\norder reduced (both)", verdict_color=aqua)
draw_row!(ax, -6, [4]; data=centers, checked=centers,
          label="center values (tracers, u in y)\nmain and this PR: check centers", verdict="solid center found:\norder reduced (unchanged)", verdict_color=aqua)

Legend(fig[2, 1], [MarkerElement(marker=:circle, color=blue, markersize=13),
                   MarkerElement(marker=:rect, color=:transparent, strokecolor=ink, strokewidth=1.5, markersize=13),
                   MarkerElement(marker=:xcross, color=red, markersize=15),
                   PolyElement(color=solid, strokecolor=ink2, strokewidth=1)],
       ["stencil data", "node checked with inactive_node", "inactive node found", "immersed cell"],
       orientation=:horizontal, framevisible=false, tellheight=true)
save("stencil_check.png", fig, px_per_unit=2)
