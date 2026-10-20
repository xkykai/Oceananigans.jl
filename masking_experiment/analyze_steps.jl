using Serialization
using Printf
using CairoMakie

U, N² = 0.1, 1.0
load(build, bottom, N) = deserialize("steps_$build/steps_$(bottom)_$N.jls")

function perturbations(run, t)
    s = run.snapshots[t]
    u = (s.u .+ circshift(s.u, (-1, 0))) ./ 2 .- U
    w = (s.w[:, 1:end-1] .+ s.w[:, 2:end]) ./ 2
    b = s.b .- N² .* run.z'
    return (; u, w, b)
end

function coarsen(q, V, r)
    N = size(q, 1) ÷ r
    qᶜ = zeros(N, N)
    Vᶜ = zeros(N, N)
    for I in 1:N, K in 1:N
        block = (I-1)*r+1:I*r, (K-1)*r+1:K*r
        Vᶜ[I, K] = sum(V[block...])
        qᶜ[I, K] = Vᶜ[I, K] > 0 ? sum(V[block...] .* q[block...]) / Vᶜ[I, K] : 0.0
    end
    return qᶜ, Vᶜ
end

function relative_errors(run, reference, t)
    q, qʳ = perturbations(run, t), perturbations(reference, t)
    r = reference.N ÷ run.N
    errors = map((:u, :w, :b)) do name
        qᶜ, Vᶜ = coarsen(getproperty(qʳ, name), reference.volume, r)
        V = run.volume
        sqrt(sum(V .* (getproperty(q, name) .- qᶜ) .^ 2) / sum(V .* qᶜ .^ 2))
    end
    return NamedTuple{(:u, :w, :b)}(errors)
end

Ns = (64, 128, 256, 512)
results = Dict()
for bottom in ("full", "partial"), reference_build in ("base", "pr")
    reference = load(reference_build, bottom, 1024)
    println("== $bottom cells, reference: 1024² from ", reference_build == "base" ? "main" : "PR")
    for t in (2.0, 4.0), N in Ns, build in ("base", "pr")
        e = relative_errors(load(build, bottom, N), reference, t)
        results[(bottom, reference_build, t, N, build)] = e
        @printf("  t=%.0f N=%4d %-4s  u: %.3e  w: %.3e  b: %.3e\n", t, N, build == "base" ? "main" : "PR", e.u, e.w, e.b)
    end
end
serialize("steps_errors.jls", results)

ink, ink2, surface = colorant"#0b0b0b", colorant"#52514e", colorant"#fcfcfb"
blue, orange = colorant"#2a78d6", colorant"#eb6834"
set_theme!(fontsize=14, textcolor=ink, backgroundcolor=surface,
           Axis=(backgroundcolor=surface, xgridvisible=false, ygridvisible=false, topspinevisible=false,
                 rightspinevisible=false, leftspinecolor=ink2, bottomspinecolor=ink2))

fig = Figure(size=(1150, 640))
for (row, bottom) in enumerate(("full", "partial")), (col, name) in enumerate((:u, :w, :b))
    label = Dict(:u => "u − U", :w => "w", :b => "b − N²z")[name]
    ax = Axis(fig[row, col]; xscale=log2, yscale=log10, xticks=collect(Ns),
              title = row == 1 ? label : "", xlabel = row == 2 ? "N" : "",
              ylabel = col == 1 ? (bottom == "full" ? "GridFittedBottom\n" : "PartialCellBottom\n") * "relative L2 error, t = 4" : "")
    for (build, color, marker, legend) in (("base", orange, :rect, "main"), ("pr", blue, :circle, "this PR"))
        e = [getproperty(results[(bottom, "pr", 4.0, N, build)], name) for N in Ns]
        scatterlines!(ax, collect(Ns), e; color, marker, markersize=10, linewidth=2, label=legend)
    end
    row == 1 && col == 1 && axislegend(ax, position=:lb, framevisible=false)
end
Label(fig[0, 1:3], "WENOVectorInvariant over a stepped sill, error against 1024² of the same geometry", fontsize=16, tellwidth=false)
save("steps_convergence.png", fig, px_per_unit=2)
