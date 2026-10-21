using CairoMakie, Serialization
U, N² = 0.1, 1.0
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])

# directory prefix, masked build, unmasked build
schemes = ((:WENO, "flux-form WENO()", "stepsweno", "masked (main)", "unmasked (this PR)"),
           (:WENOVectorInvariant, "WENOVectorInvariant()", "steps", "masked (main, this PR)", "unmasked"))

function perturbation(run, name)
    s = run.snapshots[4.0]
    q = name == :u ? (s.u .+ circshift(s.u, (-1, 0))) ./ 2 .- U : s.b .- N² .* run.z'
    return ifelse.(run.volume .> 0, q, NaN)
end

fig = Figure(size=(1500, 1250))
columns = ((128, :masked), (128, :unmasked), (256, :masked), (256, :unmasked))
for (s_index, (_, scheme_label, directory, masked_label, unmasked_label)) in enumerate(schemes)
    for (f_index, (name, label, range)) in enumerate(((:u, "u − U", 0.12), (:b, "b − N²z", 0.06)))
        row = 2(s_index - 1) + f_index
        for (col, (N, version)) in enumerate(columns)
            build = version == :masked ? "base" : "pr"
            run = deserialize("$(directory)_$build/steps_partial_$N.jls")
            x = z = ((1:N) .- 0.5) ./ N
            ax = Axis(fig[row, col]; title = "$(N)², " * (version == :masked ? masked_label : unmasked_label), titlesize=13,
                      aspect=DataAspect(), limits=((0.15, 0.9), (0.0, 0.65)),
                      xlabel = row == 4 ? "x" : "", ylabel = col == 1 ? "$scheme_label\n$label\nz" : "",
                      yticklabelsvisible = col == 1, xticklabelsvisible = row == 4)
            heatmap!(ax, x, z, perturbation(run, name); colormap=diverging, colorrange=(-range, range), nan_color=solid)
        end
        Colorbar(fig[row, 5]; colormap=diverging, colorrange=(-range, range), label=label, height=Relative(0.8))
    end
end
Label(fig[0, 1:4], "Stratified flow over a stepped sill, PartialCellBottom, t = 4: masked and unmasked immersed momentum fluxes",
      fontsize=16, tellwidth=false)
save("sill_side_by_side.png", fig, px_per_unit=2)
