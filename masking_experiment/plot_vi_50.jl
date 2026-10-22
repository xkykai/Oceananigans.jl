using CairoMakie, Serialization, Printf
U, N², L, H = 0.2, 1e-6, 20e3, 2e3
ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
solid = colorant"#d9d8d4"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))
diverging = cgrad([colorant"#104281", colorant"#5598e7", colorant"#f0efec", colorant"#ec835a", colorant"#9a2f0e"])

perturbation(r, name) = ifelse.(r.volume .> 0, name == :u ? (r.u .+ circshift(r.u, (-1, 0))) ./ 2 .- U : r.b .- N² .* r.z', NaN)
function coarsen(q, V, r)
    n = size(q, 1) ÷ r
    [(Vb = V[(I-1)*r+1:I*r, (K-1)*r+1:K*r]; s = sum(Vb); s > 0 ? sum(Vb .* replace(q[(I-1)*r+1:I*r, (K-1)*r+1:K*r], NaN => 0)) / s : NaN) for I in 1:n, K in 1:n]
end

N = 50
truth = deserialize("leenh_6187/lee_nonhydrostatic_400.jls")
masked, unmasked = deserialize("leeconv_base/lee_WENOVectorInvariant_$N.jls"), deserialize("leeconv_pr/lee_WENOVectorInvariant_$N.jls")
V = masked.volume .* [abs(-L + (i - 0.5) * 2L / N) < L / 2 for i in 1:N, k in 1:N]
relative(e, q) = (ok = .!isnan.(e) .& .!isnan.(q) .& (V .> 0); sqrt(sum(V[ok] .* e[ok] .^ 2) / sum(V[ok] .* q[ok] .^ 2)))

x(n) = (-L .+ ((1:n) .- 0.5) .* 2L / n) ./ 1e3
z(n) = (-H .+ ((1:n) .- 0.5) .* H / n) ./ 1e3
fig = Figure(size=(2000, 640))
for (row, (name, label, range, unit)) in enumerate(((:u, "u − U", 0.15, "m/s"), (:b, "b − N²z", 1.5e-4, "m/s²")))
    qᵗ = coarsen(perturbation(truth, name), truth.volume, 400 ÷ N)
    qᵐ, qᵘ = perturbation(masked, name), perturbation(unmasked, name)
    panels = (("truth: nonhydrostatic, 400²", perturbation(truth, name), 400, range),
              ("WENOVectorInvariant(), masked, 50²", qᵐ, N, range),
              ("WENOVectorInvariant(), unmasked, 50²", qᵘ, N, range),
              (@sprintf("masked − truth (error %.2f)", relative(qᵐ .- qᵗ, qᵗ)), qᵐ .- qᵗ, N, range / 3),
              (@sprintf("unmasked − truth (error %.2f)", relative(qᵘ .- qᵗ, qᵗ)), qᵘ .- qᵗ, N, range / 3),
              ("masked − unmasked", qᵐ .- qᵘ, N, range / 3))
    for (col, (title, q, n, crange)) in enumerate(panels)
        ax = Axis(fig[row, col]; title, titlesize=13, xlabel = row == 2 ? "x (km)" : "", ylabel = col == 1 ? "$label\nz (km)" : "",
                  yticklabelsvisible = col == 1, xticklabelsvisible = row == 2, limits=((-18, 2), (-2, 0)))
        heatmap!(ax, x(n), z(n), q; colormap=diverging, colorrange=(-crange, crange), nan_color=solid)
    end
    Colorbar(fig[row, 7]; colormap=diverging, colorrange=(-range, range), label="$label ($unit)", height=Relative(0.8))
    Colorbar(fig[row, 8]; colormap=diverging, colorrange=(-range / 3, range / 3), label="difference", height=Relative(0.8))
end
Label(fig[0, 1:6], "Lee wave from #5693, day 15: WENOVectorInvariant() at 50² with and without the masking, against the nonhydrostatic 400² truth",
      fontsize=16, tellwidth=false)
save("lee_vector_invariant_50.png", fig, px_per_unit=2)
