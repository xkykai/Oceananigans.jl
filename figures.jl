using Oceananigans
using Oceananigans.Advection: div_𝐯u, outside_symmetric_halo_xᶜ, outside_symmetric_halo_xᶠ, HOADV
using Oceananigans.Grids: required_halo_size_x
using CairoMakie
using Printf

outdir = joinpath(@__DIR__, "figs")
mkpath(outdir)

N = 32
grid = RectilinearGrid(size=(N, N), halo=(5, 5), x=(0, 1), z=(0, 1), topology=(Bounded, Flat, Bounded))

function tendency(advection, φwall)
    model = NonhydrostaticModel(grid; advection)
    set!(model, u = (x, z) -> 1 + sin(3x) * cos(2π * z), w = (x, z) -> sin(4π * x) * cos(π * z))
    φ = XFaceField(grid)
    set!(φ, 1)
    interior(φ, 1, 1, :) .= φwall
    interior(φ, N+1, 1, :) .= φwall
    G = Field(KernelFunctionOperation{Face, Center, Center}(div_𝐯u, grid, model.advection.momentum, model.velocities, φ))
    compute!(G)
    return Array(interior(G, :, 1, :))
end

function transport_order(i, scheme, outside)
    scheme isa HOADV || return 2 * required_halo_size_x(scheme)
    outside(i, Bounded, N, scheme) && return 2 * required_halo_size_x(scheme.advecting_velocity_scheme)
    return transport_order(i, scheme.buffer_scheme, outside)
end

schemes = [("Centered(order=2)", Centered()), ("WENO(order=5)", WENO()), ("WENO(order=9)", WENO(order=9))]
colors  = ["#2a78d6", "#eb6834", "#1baf7a"]
text_secondary = "#52514e"
ramp = cgrad(["#f4f7fb", "#9cc1ec", "#2a78d6", "#0d2f5c"])

admissible = [tendency(s, 0) for (_, s) in schemes]
uniform    = [tendency(s, 1) for (_, s) in schemes]

column_max(G) = [maximum(abs, G[i, :]) for i in 1:N+1]

for ((name, scheme), Gᵃ, Gᵘ) in zip(schemes, admissible, uniform)
    println(name)
    for i in 2:8
        pᴸ = transport_order(i - 1, scheme, outside_symmetric_halo_xᶜ)
        pᴿ = transport_order(i, scheme, outside_symmetric_halo_xᶜ)
        pʷ = transport_order(i, scheme, outside_symmetric_halo_xᶠ)
        @printf("  i=%d  ũ orders (%d, %d)  w̃ order %d   max|G| admissible %.1e  uniform %.1e\n",
                i, pᴸ, pᴿ, pʷ, column_max(Gᵃ)[i], column_max(Gᵘ)[i])
    end
end

#####
##### Static figure
#####

fig = Figure(size=(1250, 860), fontsize=16, backgroundcolor=:white)

Label(fig[0, 1:3], "Advective tendency of φ by a divergence-free flow, φ = 0 on the walls and 1 elsewhere",
      fontsize=19, font=:bold, tellwidth=false)

hm = nothing
for (n, ((name, _), Gᵃ)) in enumerate(zip(schemes, admissible))
    ax = Axis(fig[1, n]; title=name, xlabel="face index i", ylabel = n == 1 ? "cell index k" : "",
              aspect=DataAspect(), yticklabelsvisible = n == 1)
    data = log10.(max.(abs.(Gᵃ), 1e-16))
    data[[1, 2, N, N+1], :] .= NaN
    global hm = heatmap!(ax, 1:N+1, 1:N, data; colormap=ramp, colorrange=(-14, 0), lowclip="#f4f7fb", nan_color="#d9d8d4")
    xlims!(ax, 0.5, N + 1.5)
end
Colorbar(fig[1, 4], hm; label="log₁₀ |G|", height=Relative(0.8))

ax = Axis(fig[2, 1:3]; yscale=log10, xlabel="face index i", ylabel="maxₖ |G|",
          title="Column maximum of |G|: filled = φ zero on walls, open = φ uniform (identical for i ≥ 3)",
          xticks=3:2:N-1, limits=((2.5, N - 0.5), (1e-15, 10)))
for ((name, _), Gᵃ, Gᵘ, color) in zip(schemes, admissible, uniform, colors)
    i = 3:N-1
    lines!(ax, i, column_max(Gᵃ)[i]; color, linewidth=2)
    scatter!(ax, i, column_max(Gᵃ)[i]; color, markersize=9, label=name)
    scatter!(ax, i, column_max(Gᵘ)[i]; color=:transparent, strokecolor=color, strokewidth=1.5, markersize=17)
end
axislegend(ax; position=:ct, orientation=:horizontal, framevisible=false)
Label(fig[3, 1:3], "Gray columns (i = 1, 2, N, N+1) read the wall value and are excluded. Away from the walls every scheme gives round-off.",
      fontsize=14, color=text_secondary, tellwidth=false)
rowsize!(fig.layout, 2, Relative(0.32))

save(joinpath(outdir, "tendency_near_walls.png"), fig, px_per_unit=2)

#####
##### Animation: transport orders around the u-cell, WENO(order=5)
#####

scheme = WENO()
Gᵃ = admissible[2]
order_color(p) = p == 4 ? colors[1] : colors[2]
ũ_stencil(c, p) = (c - p ÷ 2 + 1):(c + p ÷ 2)     # u faces read by the transport at center c
w̃_stencil(f, p) = (f - p ÷ 2):(f + p ÷ 2 - 1)     # w centers read by the transport at face f

anim = Figure(size=(1100, 600), fontsize=16, backgroundcolor=:white)
title = Observable("")
Label(anim[0, 1], title; fontsize=19, font=:bold, tellwidth=false)
subtitle = Observable("")
Label(anim[1, 1], subtitle; fontsize=16, color=text_secondary, tellwidth=false)

axs = Axis(anim[2, 1]; aspect=DataAspect(), limits=((0.2, 13.8), (-1.65, 1.45)))
hidedecorations!(axs)
hidespines!(axs)

axb = Axis(anim[3, 1]; yscale=log10, xlabel="face index i", ylabel="maxₖ |G|", xticks=2:9,
           limits=((1.4, 9.6), (1e-15, 10)))

function draw_frame!(i)
    empty!(axs)
    empty!(axb)

    pᴸ = transport_order(i - 1, scheme, outside_symmetric_halo_xᶜ)
    pᴿ = transport_order(i, scheme, outside_symmetric_halo_xᶜ)
    pʷ = transport_order(i, scheme, outside_symmetric_halo_xᶠ)
    consistent = pᴸ == pᴿ == pʷ

    title[] = "WENO(order=5): transports around the u-cell at face i = $i"
    subtitle[] = consistent ? "all transports order $pᴸ → they cancel, uniform φ is preserved" :
                              "ũ orders ($pᴸ, $pᴿ), w̃ order $pʷ → mixed orders, transports do not cancel"

    poly!(axs, Rect(0.2, -0.1, 0.8, 1.2); color="#d9d8d4")
    text!(axs, 0.6, 1.25; text="wall", align=(:center, :bottom), color=text_secondary, fontsize=14)
    for f in 1:10
        lines!(axs, [f, f], [0, 1]; color = f == 1 ? :black : "#bdbcb7", linewidth = f == 1 ? 3 : 1)
    end
    lines!(axs, [1, 10], [0, 0]; color="#bdbcb7")
    lines!(axs, [1, 10], [1, 1]; color="#bdbcb7")

    poly!(axs, Rect(i - 0.5, 0, 1, 1); color=(:black, 0.06), strokecolor=:black, strokewidth=1.5, linestyle=:dash)
    scatter!(axs, 1:10, fill(0.5, 10); color="#8a8984", markersize=8)
    scatter!(axs, (1:9) .+ 0.5, zeros(9); color="#8a8984", marker=:utriangle, markersize=10)
    scatter!(axs, (1:9) .+ 0.5, ones(9); color="#8a8984", marker=:utriangle, markersize=10)

    arrows2d!(axs, [Point2f(i - 0.5 - 0.2, 0.5), Point2f(i + 0.5 - 0.2, 0.5)], [Vec2f(0.4, 0), Vec2f(0.4, 0)];
              color=[order_color(pᴸ), order_color(pᴿ)], shaftwidth=4, tipwidth=14, tiplength=12)
    arrows2d!(axs, [Point2f(i, -0.2), Point2f(i, 0.8)], [Vec2f(0, 0.4), Vec2f(0, 0.4)];
              color=order_color(pʷ), shaftwidth=4, tipwidth=14, tiplength=12)

    rows = [("ũ at center i-1, order $pᴸ", ũ_stencil(i - 1, pᴸ), pᴸ, -0.5),
            ("ũ at center i, order $pᴿ",   ũ_stencil(i, pᴿ) ,   pᴿ, -0.95),
            ("w̃ at face i, order $pʷ",     w̃_stencil(i, pʷ) .+ 0.5, pʷ, -1.4)]
    for (label, xs, p, z) in rows
        lines!(axs, [first(xs), last(xs)], [z, z]; color=order_color(p), linewidth=2)
        scatter!(axs, collect(xs), fill(z, length(xs)); color=order_color(p), markersize=11)
        text!(axs, 10.3, z; text=label, align=(:left, :center), fontsize=14, color=:black)
    end
    text!(axs, 10.3, 0.5; text="● u    ▲ w", align=(:left, :center), fontsize=14, color=text_secondary)
    scatter!(axs, [10.45, 12.05], [1.25, 1.25]; color=[colors[1], colors[2]], marker=:rect, markersize=12)
    text!(axs, 10.65, 1.25; text="order 4", align=(:left, :center), fontsize=14, color=:black)
    text!(axs, 12.25, 1.25; text="order 2", align=(:left, :center), fontsize=14, color=:black)
    text!(axs, 10.3, -0.12; text="stencils:", align=(:left, :center), fontsize=14, color=text_secondary)

    is = 2:9
    G = column_max(Gᵃ)[is]
    barcolors = [j == 2 ? "#e8e7e3" : j == i ? "#0b0b0b" : "#a3a29d" for j in is]
    barplot!(axb, is, G; color=barcolors, fillto=1e-15, gap=0.25)
    text!(axb, 2, 2e-15; text="reads wall", align=(:center, :bottom), fontsize=12, color=text_secondary)
    return nothing
end

rowsize!(anim.layout, 3, Relative(0.32))
rowgap!(anim.layout, 4)

frames = repeat(2:8, inner=3)
record(anim, joinpath(outdir, "transport_orders_weno5.gif"), frames; framerate=2) do i
    draw_frame!(i)
end

draw_frame!(3)
save(joinpath(outdir, "transport_orders_weno5_i3.png"), anim, px_per_unit=2)
