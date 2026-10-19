using CairoMakie, Serialization, Printf
U = 0.2
load(b, s) = deserialize("lee_$b/lee_$s.jls")
names = ("WENO", "WENOVectorInvariant", "VectorInvariant")
rms(a) = sqrt(sum(a .^ 2) / length(a))
for b in ("base", "pr")
    r = Dict(s => load(b, s).u .- U for s in names)
    @printf("%-4s  rms(WENO − VI) = %.4f  rms(WENOVI − VI) = %.4f  rms(WENO − WENOVI) = %.4f\n", b,
            rms(r["WENO"] .- r["VectorInvariant"]), rms(r["WENOVectorInvariant"] .- r["VectorInvariant"]), rms(r["WENO"] .- r["WENOVectorInvariant"]))
end
ink2, surface = colorant"#52514e", colorant"#fcfcfb"
set_theme!(fontsize=14, backgroundcolor=surface, Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false,
           topspinevisible=false, rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
colormap = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
fig = Figure(size=(1300, 560))
titles = Dict("WENO" => "WENO()", "WENOVectorInvariant" => "WENOVectorInvariant()", "VectorInvariant" => "VectorInvariant()")
for (row, (b, label)) in enumerate((("base", "main"), ("pr", "this PR"))), (col, s) in enumerate(names)
    r = load(b, s)
    ax = Axis(fig[row, col]; title = row == 1 ? titles[s] : "", ylabel = col == 1 ? "$label\nz (km)" : "",
              xlabel = row == 2 ? "x (km)" : "", xticklabelsvisible = row == 2, yticklabelsvisible = col == 1)
    heatmap!(ax, r.x[1:end] ./ 1e3, r.z ./ 1e3, r.u .- U; colormap, colorrange=(-0.15, 0.15))
    xlims!(ax, -18, 2)
end
Colorbar(fig[1:2, 4]; colormap, colorrange=(-0.15, 0.15), label="u − U (m/s)")
Label(fig[0, 1:3], "Lee wave from #5693 at day 15 (HydrostaticFreeSurfaceModel, PartialCellBottom)", fontsize=16, tellwidth=false)
save("lee_wave_schemes.png", fig, px_per_unit=2)
