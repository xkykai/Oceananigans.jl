using CairoMakie, Serialization, Printf
U, N², L, H = 0.2, 1e-6, 20e3, 2e3
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
errors = deserialize("errors_vs_nonhydrostatic.jls")

function perturbation(r, name)
    q = name == :u ? (r.u .+ circshift(r.u, (-1, 0))) ./ 2 .- U : r.b .- N² .* r.z'
    return ifelse.(r.volume .> 0, q, NaN)
end

panels = (("truth: nonhydrostatic WENO(),\nunmasked, 400²", "leenh_6187/lee_nonhydrostatic_400.jls", nothing),
          ("nonhydrostatic WENO(),\nunmasked (this PR)", "leenh_6187/lee_nonhydrostatic_100.jls", "nonhydrostatic WENO, unmasked (this PR)"),
          ("hydrostatic WENO(),\nunmasked (this PR)", "leeconv_6187/lee_WENO_100.jls", "hydrostatic WENO, unmasked (this PR)"),
          ("hydrostatic WENO(),\nmasked (main)", "leeconv_base/lee_WENO_100.jls", "hydrostatic WENO, masked (main)"),
          ("hydrostatic WENOVectorInvariant(),\nmasked (main, this PR)", "leeconv_base/lee_WENOVectorInvariant_100.jls", "hydrostatic WENOVectorInvariant, masked"),
          ("hydrostatic WENOVectorInvariant(),\nunmasked", "leeconv_pr/lee_WENOVectorInvariant_100.jls", "hydrostatic WENOVectorInvariant, unmasked"))

fig = Figure(size=(2000, 640))
for (row, (name, label, range, unit)) in enumerate(((:u, "u − U", 0.15, "m/s"), (:b, "b − N²z", 1.5e-4, "m/s²")))
    for (col, (title, file, key)) in enumerate(panels)
        r = deserialize(file)
        N = r.N
        x = (-L .+ ((1:N) .- 0.5) .* 2L / N) ./ 1e3
        z = (-H .+ ((1:N) .- 0.5) .* H / N) ./ 1e3
        e = isnothing(key) ? "" : @sprintf("\nerror u %.2f, b %.2f", errors[(key, 100)]...)
        ax = Axis(fig[row, col]; title = row == 1 ? title * (N == 100 ? ", 100²" : "") * e : "", titlesize=13,
                  xlabel = row == 2 ? "x (km)" : "", ylabel = col == 1 ? "$label\nz (km)" : "",
                  yticklabelsvisible = col == 1, xticklabelsvisible = row == 2, limits=((-18, 2), (-2, 0)))
        heatmap!(ax, x, z, perturbation(r, name); colormap=diverging, colorrange=(-range, range), nan_color=solid)
    end
    Colorbar(fig[row, 7]; colormap=diverging, colorrange=(-range, range), label="$label ($unit)", height=Relative(0.8))
end
Label(fig[0, 1:6], "Lee wave from #5693, day 15: nonhydrostatic 400² truth and runs at 100² (error: relative L2 against the truth for |x| < L/2)",
      fontsize=16, tellwidth=false)
save("lee_vs_nonhydrostatic.png", fig, px_per_unit=2)
