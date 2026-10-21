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

results = Dict()
for scheme in (:WENO, :WENOVectorInvariant), reference_version in (:masked, :unmasked)
    reference = load(getproperty(builds[scheme], reference_version), scheme, 400)
    any(isnan, reference.u) && (println("== $scheme, reference: 400² $reference_version blew up"); continue)
    println("== $scheme, reference: 400² $reference_version")
    for N in Ns, version in (:masked, :unmasked)
        run = load(getproperty(builds[scheme], version), scheme, N)
        e = any(isnan, run.u) ? (u = NaN, w = NaN, b = NaN) : relative_error(run, reference)
        results[(scheme, reference_version, N, version)] = e
        @printf("  N=%3d %-8s  u: %.3e  w: %.3e  b: %.3e\n", N, version, e.u, e.w, e.b)
    end
end
serialize("lee_convergence_errors.jls", results)

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
blue, orange = colorant"#2a78d6", colorant"#eb6834"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))

fig = Figure(size=(1200, 640))
labels = (WENO = ("flux-form WENO()", "masked (main)", "unmasked (this PR)"),
          WENOVectorInvariant = ("WENOVectorInvariant()", "masked (main, this PR)", "unmasked"))
for (row, scheme) in enumerate((:WENO, :WENOVectorInvariant)), (col, name) in enumerate((:u, :w, :b))
    title, masked_label, unmasked_label = labels[scheme]
    ax = Axis(fig[row, col]; xscale=log2, yscale=log10, xticks=collect(Ns),
              title = row == 1 ? Dict(:u => "u − U", :w => "w", :b => "b − N²z")[name] : "",
              xlabel = row == 2 ? "N" : "", ylabel = col == 1 ? "$title\nrelative error, day 15" : "")
    for (version, color, marker, legend) in ((:masked, orange, :rect, masked_label), (:unmasked, blue, :circle, unmasked_label))
        for (reference_version, style) in ((:unmasked, :solid), (:masked, :dash))
            haskey(results, (scheme, reference_version, first(Ns), version)) || continue
            e = [getproperty(results[(scheme, reference_version, N, version)], name) for N in Ns]
            scatterlines!(ax, collect(Ns), e; color, marker, markersize=9, linewidth=2, linestyle=style,
                          label = reference_version == :unmasked ? legend : nothing)
        end
    end
    col == 1 && axislegend(ax, position=:lb, framevisible=false)
end
Label(fig[0, 1:3], "Lee wave from #5693, error against 400² for |x| < L/2 (solid: unmasked reference, dashed: masked reference)",
      fontsize=16, tellwidth=false)
save("lee_convergence.png", fig, px_per_unit=2)

diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])
for scheme in (:WENO, :WENOVectorInvariant)
    title, masked_label, unmasked_label = labels[scheme]
    N = 100
    m, u = load(builds[scheme].masked, scheme, N), load(builds[scheme].unmasked, scheme, N)
    references = filter(r -> !any(isnan, r.u), [load(builds[scheme].masked, scheme, 400), load(builds[scheme].unmasked, scheme, 400)])
    qʳ = sum(coarsen(perturbations(r).u, r.volume, 4) for r in references) ./ length(references)
    qᵐ, qᵘ = perturbations(m).u, perturbations(u).u
    x = (-L .+ ((1:N) .- 0.5) .* 2L / N) ./ 1e3
    z = (-H .+ ((1:N) .- 0.5) .* H / N) ./ 1e3
    fig = Figure(size=(1700, 420))
    for (col, (panel, q, range)) in enumerate((("reference (400²)", qʳ, 0.15), (masked_label, qᵐ, 0.15), (unmasked_label, qᵘ, 0.15),
                                               ("error, " * masked_label, qᵐ .- qʳ, 0.03), ("error, " * unmasked_label, qᵘ .- qʳ, 0.03)))
        ax = Axis(fig[1, col]; title=panel, xlabel="x (km)", ylabel = col == 1 ? "z (km)" : "", yticklabelsvisible = col == 1,
                  limits=((-18, 2), (-2, 0)))
        heatmap!(ax, x, z, q; colormap=diverging, colorrange=(-range, range), nan_color=solid)
    end
    Colorbar(fig[1, 6]; colormap=diverging, colorrange=(-0.15, 0.15), label="u − U (m/s)")
    Colorbar(fig[1, 7]; colormap=diverging, colorrange=(-0.03, 0.03), label="error (m/s)")
    Label(fig[0, 1:5], "$title, lee wave from #5693 at 100², day 15 (reference: 400² runs that did not blow up, averaged to 100²)",
          fontsize=16, tellwidth=false)
    save("lee_fields_$(scheme).png", fig, px_per_unit=2)
end
