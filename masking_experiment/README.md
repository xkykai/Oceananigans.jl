Does keeping the immersed momentum flux masking matter for https://github.com/CliMA/Oceananigans.jl/pull/6187#discussion_r4310849682?

Three checkouts: the PR merge base (`2667412`), the PR head (`fc6d37c`), and the PR head with
`src/Advection/immersed_advective_fluxes.jl` restored from the merge base ("PR + masking").

    julia --project=<checkout> continuity.jl <outdir>        # momentum-cell continuity and face audit
    julia --project=<checkout> nonimmersed.jl                # same residual on a grid without immersed boundary
    julia --project=<checkout> animate_runs.jl anim <case>   # case ∈ masked, pr, weno_c2
    julia plot_continuity.jl                                 # reads cont_masked/, cont_pr/
    julia plot_animation.jl                                  # reads anim/

`momentum_cell_continuity.png`: net advecting volume flux of each momentum cell for a projected flow over a 64 × 64 staircase.
`immersed_face_transport.png`: transport used by the momentum flux on the faces the masking zeroes, against the volume crossing their open part.
`staircase_convection_masking.{mp4,gif}`: 128 × 128 partial-cell convection with `WENO()`, PR + masking vs PR.
