using CairoMakie, Serialization

E = ARGS[1]
load(dir, name) = deserialize(joinpath(E, dir, name * ".jls"))
colors = Dict("main" => :firebrick, "this PR" => :royalblue)
ylimits = (0.5, 1e33)

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1500, 550))
panels = (("GridFittedBottom", "ke_full"), ("PartialCellBottom", "ke_partial"))

for (n, (title, name)) in enumerate(panels)
    ax = Axis(fig[1, n]; title="Inviscid Centered() flow over a $title staircase", xlabel="t",
              ylabel = n == 1 ? "KE(t) / KE(0)" : "", yscale=log10)
    for (label, dir) in (("main", "final_base"), ("this PR", "final_pr")),
        (case, style, suffix) in ((name, :solid, ""), (name * "_half_dt", :dot, ", Δt / 2"))

        r = load(dir, case)
        ratio = r.E ./ r.E[1]
        keep = isfinite.(ratio)
        lines!(ax, r.t[keep], ratio[keep]; color=colors[label], linestyle=style, linewidth=3, label=label * suffix)

        i = findfirst(x -> !isfinite(x) || x > ylimits[2], ratio)
        if !isnothing(i) && style == :solid
            text!(ax, r.t[i], ylimits[2]; text=" blows up", color=colors[label], align=(:left, :top))
        end
    end
    hlines!(ax, 1; color=:gray, linewidth=1)
    xlims!(ax, 0, 4)
    ylims!(ax, ylimits...)
    n == 1 && axislegend(ax; position=:lt)
end

save(joinpath(E, "kinetic_energy.png"), fig)
