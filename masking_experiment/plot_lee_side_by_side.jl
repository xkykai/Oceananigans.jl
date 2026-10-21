using CairoMakie, Serialization
U, N², L, H = 0.2, 1e-6, 20e3, 2e3
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])

load(build, scheme, N) = deserialize("leeconv_$build/lee_$(scheme)_$N.jls")
builds = (WENO = (masked = "base", unmasked = "6187"), WENOVectorInvariant = (masked = "base", unmasked = "pr"))
version_labels = (WENO = (masked = "masked (main)", unmasked = "unmasked (this PR)"),
                  WENOVectorInvariant = (masked = "masked (main, this PR)", unmasked = "unmasked"))

function perturbation(run, name)
    q = name == :u ? (run.u .+ circshift(run.u, (-1, 0))) ./ 2 .- U : run.b .- N² .* run.z'
    return ifelse.(run.volume .> 0, q, NaN)
end

fig = Figure(size=(1500, 1250))
columns = ((50, :masked), (50, :unmasked), (100, :masked), (100, :unmasked))
for (s_index, (scheme, scheme_label)) in enumerate(((:WENO, "flux-form WENO()"), (:WENOVectorInvariant, "WENOVectorInvariant()")))
    for (f_index, (name, label, range, unit)) in enumerate(((:u, "u − U", 0.15, "m/s"), (:b, "b − N²z", 1.5e-4, "m/s²")))
        row = 2(s_index - 1) + f_index
        for (col, (N, version)) in enumerate(columns)
            run = load(getproperty(builds[scheme], version), scheme, N)
            x = (-L .+ ((1:N) .- 0.5) .* 2L / N) ./ 1e3
            z = (-H .+ ((1:N) .- 0.5) .* H / N) ./ 1e3
            ax = Axis(fig[row, col]; title = "$(N)², " * getproperty(version_labels[scheme], version), titlesize=13,
                      xlabel = row == 4 ? "x (km)" : "", ylabel = col == 1 ? "$scheme_label\n$label\nz (km)" : "",
                      yticklabelsvisible = col == 1, xticklabelsvisible = row == 4, limits=((-18, 2), (-2, 0)))
            heatmap!(ax, x, z, perturbation(run, name); colormap=diverging, colorrange=(-range, range), nan_color=solid)
        end
        Colorbar(fig[row, 5]; colormap=diverging, colorrange=(-range, range), label="$label ($unit)", height=Relative(0.8))
    end
end
Label(fig[0, 1:4], "Lee wave from #5693, day 15: masked and unmasked immersed momentum fluxes at 50² and 100²", fontsize=16, tellwidth=false)
save("lee_side_by_side.png", fig, px_per_unit=2)
