using CairoMakie, Serialization, Printf

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])

U, N² = 0.1, 1.0
t, N, bottom = 4.0, 256, "partial"

function perturbations(run)
    s = run.snapshots[t]
    u = (s.u .+ circshift(s.u, (-1, 0))) ./ 2 .- U
    b = s.b .- N² .* run.z'
    fluid = run.volume .> 0
    return (; u = ifelse.(fluid, u, NaN), b = ifelse.(fluid, b, NaN), volume = run.volume)
end

function coarsen(q, V, r)
    n = size(q, 1) ÷ r
    return [(Vb = V[(I-1)*r+1:I*r, (K-1)*r+1:K*r]; s = sum(Vb);
             s > 0 ? sum(Vb .* replace(q[(I-1)*r+1:I*r, (K-1)*r+1:K*r], NaN => 0)) / s : NaN) for I in 1:n, K in 1:n]
end

function sill_fields(directory)
    ref_main = perturbations(deserialize("$(directory)_base/steps_$(bottom)_1024.jls"))
    ref_pr   = perturbations(deserialize("$(directory)_pr/steps_$(bottom)_1024.jls"))
    r = 1024 ÷ N
    reference = map((:u, :b)) do name
        (coarsen(getproperty(ref_main, name), ref_main.volume, r) .+ coarsen(getproperty(ref_pr, name), ref_pr.volume, r)) ./ 2
    end
    reference = NamedTuple{(:u, :b)}(reference)
    main = perturbations(deserialize("$(directory)_base/steps_$(bottom)_$N.jls"))
    pr   = perturbations(deserialize("$(directory)_pr/steps_$(bottom)_$N.jls"))
    return (; reference, main, pr)
end

x = ((1:N) .- 0.5) ./ N
z = ((1:N) .- 0.5) ./ N
xlims, zlims = (0.15, 0.9), (0.0, 0.65)

function relative_error(q, qʳ, V)
    fluid = .!isnan.(q) .& .!isnan.(qʳ)
    return sqrt(sum(V[fluid] .* (q[fluid] .- qʳ[fluid]) .^ 2) / sum(V[fluid] .* qʳ[fluid] .^ 2))
end

function sill_figure(directory, scheme, filename, masked_label, unmasked_label)
    F = sill_fields(directory)
    fig = Figure(size=(1700, 640))
    for (row, (name, label, range)) in enumerate(((:u, "u − U", 0.12), (:b, "b − N²z", 0.06)))
        qʳ = F.reference[name]
        qᵐ, qᵖ = getproperty(F.main, name), getproperty(F.pr, name)
        eᵐ, eᵖ = qᵐ .- qʳ, qᵖ .- qʳ
        erange = range / 3
        panels = (("reference (1024²)", qʳ, range, nothing),
                  (masked_label, qᵐ, range, nothing),
                  (unmasked_label, qᵖ, range, nothing),
                  ("error, " * masked_label, eᵐ, erange, relative_error(qᵐ, qʳ, F.main.volume)),
                  ("error, " * unmasked_label, eᵖ, erange, relative_error(qᵖ, qʳ, F.pr.volume)))
        for (col, (title, q, crange, e)) in enumerate(panels)
            ax = Axis(fig[row, col]; title = row == 1 ? title : "", aspect=DataAspect(), limits=(xlims, zlims),
                      ylabel = col == 1 ? "$label\nz" : "", xlabel = row == 2 ? "x" : "",
                      yticklabelsvisible = col == 1, xticklabelsvisible = row == 2)
            heatmap!(ax, x, z, q; colormap=diverging, colorrange=(-crange, crange), nan_color=solid)
            isnothing(e) || text!(ax, 0.17, 0.62, text=@sprintf("relative error %.3f", e), align=(:left, :top), color=ink, fontsize=13)
            col == 3 && Colorbar(fig[row, 6]; colormap=diverging, colorrange=(-crange, crange), height=Relative(0.8))
            col == 5 && Colorbar(fig[row, 7]; colormap=diverging, colorrange=(-crange, crange), height=Relative(0.8), label="error")
        end
    end
    Label(fig[0, 1:5], "$scheme over a stepped sill, PartialCellBottom, $(N)², t = 4 (reference: mean of the two 1024² runs, averaged to $(N)²)",
          fontsize=16, tellwidth=false)
    save(filename, fig, px_per_unit=2)
end

sill_figure("stepsweno", "Flux-form WENO()", "sill_fields_weno.png", "masked (main)", "unmasked (this PR)")
sill_figure("steps", "WENOVectorInvariant()", "sill_fields_vector_invariant.png", "masked (main, this PR)", "unmasked")
