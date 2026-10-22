using Serialization, Printf
U, N², L = 0.2, 1e-6, 20e3
truth = deserialize("leenh_6187/lee_nonhydrostatic_400.jls")
k = 200
w = (truth.w[:, k] .+ truth.w[:, k+1]) ./ 2
@printf("truth (nonhydrostatic 400², unmasked): w at z = −1 km: max |w| = %.3e, max neighbour jump = %.3e\n", maximum(abs, w), maximum(abs, diff(w)))

fields(r) = (u = (r.u .+ circshift(r.u, (-1, 0))) ./ 2 .- U, b = r.b .- N² .* r.z')
function coarsen(q, V, r)
    n = size(q, 1) ÷ r
    [(Vb = V[(I-1)*r+1:I*r, (K-1)*r+1:K*r]; s = sum(Vb); s > 0 ? sum(Vb .* q[(I-1)*r+1:I*r, (K-1)*r+1:K*r]) / s : NaN) for I in 1:n, K in 1:n]
end
T = fields(truth)
function error_against_truth(r)
    any(isnan, r.u) && return nothing
    N = r.N; f = fields(r); rr = 400 ÷ N
    V = r.volume .* [abs(-L + (i - 0.5) * 2L / N) < L / 2 for i in 1:N, k in 1:N]
    map((:u, :b)) do name
        qᵗ = coarsen(getproperty(T, name), truth.volume, rr); q = getproperty(f, name)
        ok = (V .> 0) .& .!isnan.(qᵗ)
        sqrt(sum(V[ok] .* (q[ok] .- qᵗ[ok]) .^ 2) / sum(V[ok] .* qᵗ[ok] .^ 2))
    end
end
runs = (("hydrostatic WENO, masked (main)", "leeconv_base/lee_WENO_%d.jls"),
        ("hydrostatic WENO, unmasked (this PR)", "leeconv_6187/lee_WENO_%d.jls"),
        ("hydrostatic WENOVectorInvariant, masked", "leeconv_base/lee_WENOVectorInvariant_%d.jls"),
        ("hydrostatic WENOVectorInvariant, unmasked", "leeconv_pr/lee_WENOVectorInvariant_%d.jls"),
        ("nonhydrostatic WENO, masked (main)", "leenh_base/lee_nonhydrostatic_%d.jls"),
        ("nonhydrostatic WENO, unmasked (this PR)", "leenh_6187/lee_nonhydrostatic_%d.jls"))
results = Dict()
for (label, pattern) in runs, N in (50, 100, 200)
    file = replace(pattern, "%d" => string(N))
    isfile(file) || continue
    e = error_against_truth(deserialize(file))
    results[(label, N)] = e
    isnothing(e) ? @printf("%-44s N=%3d  blew up\n", label, N) : @printf("%-44s N=%3d  u %.3f  b %.3f\n", label, N, e...)
end
serialize("errors_vs_nonhydrostatic.jls", results)
