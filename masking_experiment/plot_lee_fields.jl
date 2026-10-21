using CairoMakie, Serialization, Printf

U, N², L, H = 0.2, 1e-6, 20e3, 2e3
Ns = (50, 100, 200)
load(build, scheme, N) = deserialize("leeconv_$build/lee_$(scheme)_$N.jls")

# Masked and unmasked builds for each momentum advection scheme
builds = (WENO = (masked = "base", unmasked = "6187"),
          WENOVectorInvariant = (masked = "base", unmasked = "pr"))

function perturbations(run)
    fluid = run.volume .> 0
    u = (run.u .+ circshift(run.u, (-1, 0))) ./ 2 .- U
    w = (run.w[:, 1:end-1] .+ run.w[:, 2:end]) ./ 2
    b = run.b .- N² .* run.z'
    return map(q -> ifelse.(fluid, q, NaN), (; u, w, b))
end

function coarsen(q, V, r)
    nx, nz = size(q, 1) ÷ r, size(q, 2) ÷ r
    return [(Vb = V[(I-1)*r+1:I*r, (K-1)*r+1:K*r]; s = sum(Vb);
             s > 0 ? sum(Vb .* replace(q[(I-1)*r+1:I*r, (K-1)*r+1:K*r], NaN => 0)) / s : NaN) for I in 1:nx, K in 1:nz]
end

away_from_sponges(N) = [abs(-L + (i - 0.5) * 2L / N) < L / 2 for i in 1:N, k in 1:N]

function relative_error(run, reference)
    q, qʳ = perturbations(run), perturbations(reference)
    r = reference.N ÷ run.N
    V = run.volume .* away_from_sponges(run.N)
    errors = map((:u, :w, :b)) do name
        qᶜ = coarsen(getproperty(qʳ, name), reference.volume, r)
        e = getproperty(q, name) .- qᶜ
        fluid = .!isnan.(e)
        sqrt(sum(V[fluid] .* e[fluid] .^ 2) / sum(V[fluid] .* qᶜ[fluid] .^ 2))
    end
    return NamedTuple{(:u, :w, :b)}(errors)
end

labels = (WENO = ("flux-form WENO()", "masked (main)", "unmasked (this PR)"),
          WENOVectorInvariant = ("WENOVectorInvariant()", "masked (main, this PR)", "unmasked"))
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
for scheme in (:WENO, :WENOVectorInvariant)
    title, masked_label, unmasked_label = labels[scheme]
    N = 100
    m, u = load(builds[scheme].masked, scheme, N), load(builds[scheme].unmasked, scheme, N)
    references = filter(r -> !any(isnan, r.u), [load(builds[scheme].masked, scheme, 400), load(builds[scheme].unmasked, scheme, 400)])
    x = (-L .+ ((1:N) .- 0.5) .* 2L / N) ./ 1e3
    z = (-H .+ ((1:N) .- 0.5) .* H / N) ./ 1e3
    fig = Figure(size=(1700, 760))
    for (row, (name, label, range, unit)) in enumerate(((:u, "u − U", 0.15, "m/s"), (:b, "b − N²z", 1.5e-4, "m/s²")))
        qʳ = sum(coarsen(getproperty(perturbations(r), name), r.volume, 4) for r in references) ./ length(references)
        qᵐ, qᵘ = getproperty(perturbations(m), name), getproperty(perturbations(u), name)
        panels = (("reference (400²)", qʳ, range), (masked_label, qᵐ, range), (unmasked_label, qᵘ, range),
                  ("error, " * masked_label, qᵐ .- qʳ, range / 5), ("error, " * unmasked_label, qᵘ .- qʳ, range / 5))
        for (col, (panel, q, crange)) in enumerate(panels)
            ax = Axis(fig[row, col]; title = row == 1 ? panel : "", xlabel = row == 2 ? "x (km)" : "",
                      ylabel = col == 1 ? "$label\nz (km)" : "", yticklabelsvisible = col == 1, xticklabelsvisible = row == 2,
                      limits=((-18, 2), (-2, 0)))
            heatmap!(ax, x, z, q; colormap=diverging, colorrange=(-crange, crange), nan_color=solid)
        end
        Colorbar(fig[row, 6]; colormap=diverging, colorrange=(-range, range), label="$label ($unit)", height=Relative(0.8))
        Colorbar(fig[row, 7]; colormap=diverging, colorrange=(-range / 5, range / 5), label="error", height=Relative(0.8))
    end
    Label(fig[0, 1:5], "$title, lee wave from #5693 at 100², day 15 (reference: 400² runs that did not blow up, averaged to 100²)",
          fontsize=16, tellwidth=false)
    save("lee_fields_$(scheme).png", fig, px_per_unit=2)
end
