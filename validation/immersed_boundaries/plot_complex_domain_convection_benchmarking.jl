using CairoMakie
using CSV
using DataFrames

df = CSV.read("benchmark_FFT_FFTprec_noprec_MITgcmprec_report/benchmark_FFT_FFTprec_noprec_MITgcmprec_report_nvtxsum.csv", DataFrame)
@info df.Range
@info propertynames(df)

Ns = [32, 64, 128, 256]
Ns_ax = ([32, 64, 128, 256].^3) .* log.([32, 64, 128, 256])

FFTprec_ts = [df[findfirst(df.Range .== "Main:Immersed timestep, FFT preconditioner N $N"), Symbol("Med (ns)")] for N in Ns] ./ 1e9
MITgcmprec_ts = [df[findfirst(df.Range .== "Main:Immersed timestep, MITgcm preconditioner N $N"), Symbol("Med (ns)")] for N in Ns] ./ 1e9
noprec_ts = [df[findfirst(df.Range .== "Main:Immersed timestep, no preconditioner N $N"), Symbol("Med (ns)")] for N in Ns] ./ 1e9
FFT_ts = [df[findfirst(df.Range .== "Main:FFT timestep, N $N"), Symbol("Med (ns)")] for N in Ns] ./ 1e9

fig = Figure()
ax_t = Axis(fig[1, 1], xlabel="N", ylabel="Median time (s)", yscale=log10, xscale=log2, title="Sloped convection, GPU, 3D setup (N³ grid points)")
lines!(ax_t, Ns, FFTprec_ts, label="Immersed solver, FFT preconditioner")
lines!(ax_t, Ns, MITgcmprec_ts, label="Immersed solver, MITgcm preconditioner")
lines!(ax_t, Ns, noprec_ts, label="Immersed solver, No preconditioner")
lines!(ax_t, Ns, FFT_ts, label="FFT solver")
axislegend(ax_t, position=:lt)
display(fig)
# save("sloped_convection_benchmarks.png", fig, px_per_unit=4)

pcg_iter_FFTprec = [12, 12.1, 12.6, 13.25]
pcg_iter_MITgcmprec = [89, 176.4, 349.75, 685.6]
pcg_iter_noprec = [174.55, 349.55, 698.05, 1371.8]

##
fig = Figure(resolution=(960, 960), fontsize=15)
ax_t = Axis(fig[1, 1], xlabel="N³ log(N)", ylabel="Median time (s)")
scatter!(ax_t, Ns_ax, FFTprec_ts, label="Immersed solver, FFT preconditioner")
scatter!(ax_t, Ns_ax, MITgcmprec_ts, label="Immersed solver, MITgcm preconditioner")
scatter!(ax_t, Ns_ax, noprec_ts, label="Immersed solver, No preconditioner")
scatter!(ax_t, Ns_ax, FFT_ts, label="FFT solver")
lines!(ax_t, Ns_ax, FFTprec_ts)
lines!(ax_t, Ns_ax, MITgcmprec_ts)
lines!(ax_t, Ns_ax, noprec_ts)
lines!(ax_t, Ns_ax, FFT_ts)
axislegend(ax_t, position=:lt)

ax_iter = Axis(fig[1, 2], xlabel="N", ylabel="Approximate PCG iterations per timestep")
scatter!(ax_iter, Ns_ax, pcg_iter_FFTprec, label="Immersed solver, FFT preconditioner")
scatter!(ax_iter, Ns_ax, pcg_iter_MITgcmprec, label="Immersed solver, MITgcm preconditioner")
scatter!(ax_iter, Ns_ax, pcg_iter_noprec, label="Immersed solver, No preconditioner")
lines!(ax_iter, Ns_ax, pcg_iter_FFTprec)
lines!(ax_iter, Ns_ax, pcg_iter_MITgcmprec)
lines!(ax_iter, Ns_ax, pcg_iter_noprec)
axislegend(ax_iter, position=:lt)

ax_t2 = Axis(fig[2, 1], xlabel="N³ log(N)", ylabel="Δt(method) / Δt(FFT solver)")
scatter!(ax_t2, Ns_ax, FFTprec_ts ./ FFT_ts, label="Immersed solver, FFT preconditioner")
scatter!(ax_t2, Ns_ax, MITgcmprec_ts ./ FFT_ts, label="Immersed solver, MITgcm preconditioner")
scatter!(ax_t2, Ns_ax, noprec_ts ./ FFT_ts, label="Immersed solver, No preconditioner")
scatter!(ax_t2, Ns_ax, FFT_ts ./ FFT_ts, label="FFT solver")
lines!(ax_t2, Ns_ax, FFTprec_ts ./ FFT_ts)
lines!(ax_t2, Ns_ax, MITgcmprec_ts ./ FFT_ts)
lines!(ax_t2, Ns_ax, noprec_ts ./ FFT_ts)
lines!(ax_t2, Ns_ax, FFT_ts ./ FFT_ts)
# axislegend(ax_t2, position=:rc)

ax_iter2 = Axis(fig[2, 2], xlabel="N³ log(N)", ylabel="iters(method) / iters(FFT preconditioner)")
scatter!(ax_iter2, Ns_ax, pcg_iter_FFTprec ./ pcg_iter_FFTprec, label="Immersed solver, FFT preconditioner")
scatter!(ax_iter2, Ns_ax, pcg_iter_MITgcmprec ./ pcg_iter_FFTprec, label="Immersed solver, MITgcm preconditioner")
scatter!(ax_iter2, Ns_ax, pcg_iter_noprec ./ pcg_iter_FFTprec, label="Immersed solver, No preconditioner")
lines!(ax_iter2, Ns_ax, pcg_iter_FFTprec ./ pcg_iter_FFTprec)
lines!(ax_iter2, Ns_ax, pcg_iter_MITgcmprec ./ pcg_iter_FFTprec)
lines!(ax_iter2, Ns_ax, pcg_iter_noprec ./ pcg_iter_FFTprec)
# axislegend(ax_iter2, position=:lt)

supertitle = Label(fig[0, :], "Sloped convection, GPU, 3D setup (N³ grid points)", font=:bold)
# save("sloped_convection_benchmarks_MITgcmprec.png", fig, px_per_unit=4)
display(fig)

##