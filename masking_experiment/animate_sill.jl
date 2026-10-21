using CairoMakie, Serialization, Printf

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
blue, orange = colorant"#2a78d6", colorant"#eb6834"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
N², N = 1.0, 256

buoyancy_perturbation(run, t) = (b = run.snapshots[t].b .- N² .* run.z'; ifelse.(run.volume .> 0, b, NaN))

function coarsen(q, V, r)
    n = size(q, 1) ÷ r
    return [(Vb = V[(I-1)*r+1:I*r, (K-1)*r+1:K*r]; s = sum(Vb);
             s > 0 ? sum(Vb .* replace(q[(I-1)*r+1:I*r, (K-1)*r+1:K*r], NaN => 0)) / s : NaN) for I in 1:n, K in 1:n]
end

function relative_error(q, qʳ, V)
    fluid = .!isnan.(q) .& .!isnan.(qʳ)
    return sqrt(sum(V[fluid] .* (q[fluid] .- qʳ[fluid]) .^ 2) / sum(V[fluid] .* qʳ[fluid] .^ 2))
end

function animate_sill(masked_build, unmasked_build, scheme, title, masked_label, unmasked_label, filename)
    masked   = deserialize("frames_$(masked_build)_$scheme/frames_partial_$N.jls")
    unmasked = deserialize("frames_$(unmasked_build)_$scheme/frames_partial_$N.jls")
    ref₁ = deserialize("frames_$(masked_build)_$scheme/frames_partial_512.jls")
    ref₂ = deserialize("frames_$(unmasked_build)_$scheme/frames_partial_512.jls")
    times = sort(collect(keys(masked.snapshots)))
    reference(t) = (coarsen(buoyancy_perturbation(ref₁, t), ref₁.volume, 2) .+ coarsen(buoyancy_perturbation(ref₂, t), ref₂.volume, 2)) ./ 2

    references = [reference(t) for t in times]
    eᵐ = [relative_error(buoyancy_perturbation(masked, t), references[n], masked.volume) for (n, t) in enumerate(times)]
    eᵘ = [relative_error(buoyancy_perturbation(unmasked, t), references[n], unmasked.volume) for (n, t) in enumerate(times)]

    n = Observable(1)
    x = z = ((1:N) .- 0.5) ./ N
    fig = Figure(size=(1700, 620))
    panels = (("reference (512²)", () -> references[n[]], 0.06),
              (masked_label, () -> buoyancy_perturbation(masked, times[n[]]), 0.06),
              (unmasked_label, () -> buoyancy_perturbation(unmasked, times[n[]]), 0.06),
              ("error, " * masked_label, () -> buoyancy_perturbation(masked, times[n[]]) .- references[n[]], 0.02),
              ("error, " * unmasked_label, () -> buoyancy_perturbation(unmasked, times[n[]]) .- references[n[]], 0.02))
    for (col, (panel_title, field, range)) in enumerate(panels)
        ax = Axis(fig[1, col]; title=panel_title, aspect=DataAspect(), limits=((0.15, 0.9), (0.0, 0.65)),
                  xlabel="x", ylabel = col == 1 ? "z" : "", yticklabelsvisible = col == 1)
        heatmap!(ax, x, z, @lift(($n; field())); colormap=diverging, colorrange=(-range, range), nan_color=solid)
    end
    Colorbar(fig[1, 6]; colormap=diverging, colorrange=(-0.06, 0.06), label="b − N²z", height=Relative(0.8))
    Colorbar(fig[1, 7]; colormap=diverging, colorrange=(-0.02, 0.02), label="error", height=Relative(0.8))

    ax = Axis(fig[2, 2:4]; xlabel="t", ylabel="relative error in b", limits=((0, 4), (0, nothing)), height=150)
    lines!(ax, times, eᵐ; color=orange, linewidth=2, label=masked_label)
    lines!(ax, times, eᵘ; color=blue, linewidth=2, label=unmasked_label)
    vlines!(ax, @lift(times[$n]); color=ink2, linewidth=1)
    axislegend(ax, position=:lt, framevisible=false)

    Label(fig[0, 1:5], @lift(@sprintf("%s over a stepped sill, PartialCellBottom, %d², t = %.2f", title, N, times[$n])),
          fontsize=16, tellwidth=false)
    record(fig, filename * ".mp4", eachindex(times); framerate=12) do i
        n[] = i
    end
    n[] = length(times)
    save(filename * "_final.png", fig, px_per_unit=2)
    @printf("%s: final relative error masked %.3f, unmasked %.3f\n", title, eᵐ[end], eᵘ[end])
end

animate_sill("base", "6187", "WENO", "Flux-form WENO()", "masked (main)", "unmasked (this PR)", "sill_weno")
animate_sill("base", "pr", "VI", "WENOVectorInvariant()", "masked (main, this PR)", "unmasked", "sill_vector_invariant")
