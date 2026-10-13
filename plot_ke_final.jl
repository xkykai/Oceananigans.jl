using CairoMakie, Serialization

E = ARGS[1]
load(dir, name) = deserialize(joinpath(E, dir, name * ".jls"))
colors = Dict("main" => :firebrick, "this PR" => :royalblue)

set_theme!(Theme(fontsize=18))
fig = Figure(size=(1500, 600))
panels = (("GridFittedBottom", (("main", "final_base", "ke_full", "ke_full_half_dt"),
                                ("this PR", "final_pr", "ke_full", "ke_full_half_dt"))),
          ("PartialCellBottom", (("main", "final_base", "ke_partial", "ke_partial_half_dt"),
                                 ("this PR", "final_pr", "ke_partial", "ke_partial_half_dt"))))

for (n, (title, runs)) in enumerate(panels)
    ax = Axis(fig[1, n]; title="Inviscid Centered() flow over a $title staircase", xlabel="t",
              ylabel = n == 1 ? "|KE(t) / KE(0) − 1|" : "", yscale=log10)
    for (label, dir, name, name_half) in runs, (case, style, suffix) in ((name, :solid, ""), (name_half, :dot, ", Δt / 2"))
        r = load(dir, case)
        drift = abs.(r.E ./ r.E[1] .- 1)
        keep = 2:length(r.t)
        lines!(ax, r.t[keep], max.(drift[keep], 1e-9); color=colors[label], linestyle=style, linewidth=3, label=label * suffix)
    end
    xlims!(ax, 0, 4)
    ylims!(ax, 1e-8, 1)
    n == 1 && axislegend(ax; position=:rb)
end

save(joinpath(E, "kinetic_energy.png"), fig)
