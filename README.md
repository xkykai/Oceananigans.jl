Figures for https://github.com/CliMA/Oceananigans.jl/pull/6187.

Each simulation script was run with a checkout of the PR's merge base (`d6a60f2`, output `out_base`) and of the PR head (output `out_pr`):

    julia --project=<checkout> demo.jl out_<base|pr>
    julia --project=<checkout> demo_dt.jl out_<base|pr>
    julia --project=<checkout> demo2.jl out_<base|pr>

then `julia plot.jl <directory containing out_base and out_pr>` with CairoMakie.

`kinetic_energy_full_partial.png`: `demo.jl`/`demo_dt.jl` (full cells), `demo3.jl` (partial cells on `main`) and `demo4.jl` (partial cells on the PR head, output `out_prfix`), plotted with `plot_ke2.jl`.
