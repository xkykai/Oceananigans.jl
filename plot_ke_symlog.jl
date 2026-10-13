using CairoMakie, Serialization

E = ARGS[1]
load(dir, name) = deserialize(joinpath(E, dir, name * ".jls"))
colors = Dict("main" => :firebrick, "this PR" => :royalblue)

ticks = [-1e-2, -1e-4, -1e-6, 0, 1e-6, 1e-4, 1e-2, 1]
labels = ["−10⁻²", "−10⁻⁴", "−10⁻⁶", "0", "10⁻⁶", "10⁻⁴", "10⁻²", "1"]

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1500, 600))
panels = (("GridFittedBottom", "ke_full"), ("PartialCellBottom", "ke_partial"))

for (n, (title, name)) in enumerate(panels)
    ax = Axis(fig[1, n]; title="Inviscid Centered() flow over a $title staircase", xlabel="t",
              ylabel = n == 1 ? "KE(t) / KE(0) − 1" : "", yscale=Makie.Symlog10(1e-7), yticks=(ticks, labels))
    for (label, dir) in (("main", "final_base"), ("this PR", "final_pr")),
        (case, style, suffix) in ((name, :solid, ""), (name * "_half_dt", :dot, ", Δt / 2"))

        r = load(dir, case)
        drift = r.E ./ r.E[1] .- 1
        keep = isfinite.(drift) .& (drift .< 10)
        lines!(ax, r.t[keep], drift[keep]; color=colors[label], linestyle=style, linewidth=3, label=label * suffix)
        if style == :solid && !all(isfinite.(drift) .& (drift .< 10))
            text!(ax, r.t[findlast(keep)], 1; text=" blows up", color=colors[label], align=(:left, :bottom))
        end
    end
    hlines!(ax, 0; color=:gray, linewidth=1)
    xlims!(ax, 0, 4)
    ylims!(ax, -1e-2, 3)
    n == 2 && axislegend(ax; position=:rt)
end

save(joinpath(E, "kinetic_energy.png"), fig)
