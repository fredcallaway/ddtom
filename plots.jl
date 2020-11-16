using Plots
using Random
gr(label="", dpi=300)
plot([1,2])

include("model.jl")
include("figure.jl")

# %% ==================== Model illustration ====================

threshold = 1.
drift = 1.

function simulate(drift, threshold; dt=dt, maxt=100)
    x = [0.]
    for i in 1:cld(maxt,dt)
        dx = dt * drift + √dt * randn()
        push!(x, x[end]+dx)
        if !(-threshold < x[end] < threshold)
            x[end] = min(threshold, max(-threshold, x[end]))
            return x
        end
    end
    x
end
Random.seed!(5)
f = plot(grid=false, yticks=false, xticks=false, framestyle=:none, ylim=(-threshold-.01, threshold+.01))
# vline!([0], color=:black)
# hline!([-threshold, threshold], color=:black)

prms = [
    (2, "#FF6167"),
    (0.5, "#4A74FE"),
]
for (drift, color) in prms
    for i in 1:3
        plot!(simulate(drift, threshold; dt=.001); color=color, alpha=0.4)
    end
end
plot!([0, 0], [-threshold, threshold], color=:black, lw=2)
plot!([0, 1400], [threshold, threshold], color=:black, lw=2)
plot!([0, 1400], [-threshold, -threshold], color=:black, lw=2)
f
savefig("figs/ddm.pdf")
# dd = ConstDrift(drift, dt)
# bb = ConstSymBounds(threshold, dt)
# rand(sampler(dd, bb))


# %% ==================== Experiment 1 ====================


figure("exp1") do 
    rts = (3:.1:9)
    β, θ, σ = 20, 12, .4
    plot(rts, β.*MAP_drift.(rts, θ, σ),
        size=(400,300),
        xaxis="Decision Time", 
        yaxis="Model Inferred Preference", 
        ylim=(0,50),
    )
    plot!(rts, β.*posterior_mean_drift.(rts, θ, σ),
        color=:red,
        size=(400,300),
        xaxis="Decision Time", 
        yaxis="Model Inferred Preference", 
        ylim=(0,50),
    )
end



# %% ==================== DDM likelihood ====================

function plot_choice_rt(drift, threshold; max_rt=10)
    dd = ConstDrift(drift, dt)
    bb = ConstSymBounds(threshold, dt)
    pu, pl = pdf(dd, bb, max_rt)
    plot(pu)
    plot!(-pl)
    hline!([0], color=:black)
end

figure() do
    σ = .5; θ = 1.8
    plot_choice_rt(randn() * σ * √2, θ)
end
# %% ==================== Drift rate posterior ====================
trial = (2., 0)
drifts = -4:0.01:4
plot(drifts, [posterior(trial, v, 2.) for v in drifts])
vline!([MAP_drift(trial, 2.)])

# %% ==================== MAP fixed threshold ====================
rts = 0.5:.1:5
thresholds = reshape(1.:4, 1, :)
X = MAP_drift.(rts, thresholds)  # . does broadcasting, so we optimize for each combination of rt and threshold

plot(rts, X, 
    xaxis="Reaction time", 
    yaxis="MAP value difference", 
    legendtitle = "Threshold",
    label=string.(Int.(thresholds))
)
savefig("figs/MAP_fixed_threshold")

# %% ==================== MAP optimize threshold ====================
x = map(rt -> MAP_drift_threshold(rt).drift, rts)
plot(rts, x, xaxis="Reaction time", yaxis="MAP value difference")
savefig("figs/MAP_optimize_threshold")

# %% ==================== Joint posterior over drift and threshold ====================
function ticklabels(x; every=8)
    q = Int.(quantile(1:length(x), [0,.5,1]))
    eachindex(x)[q], x[q]
    # eachindex(x)[1:every:end], x[1:every:end]
end

rts = 1.:1:6
drifts = -1:0.02:4
thresholds = reshape(0.5:.1:15.5, 1, :)

figs = map(rts) do rt
    trial = (rt, 1)
    X = posterior.([trial], drifts, thresholds);

    heatmap(X',
        xaxis=("Value Difference", ticklabels(drifts)),
        yaxis=("Threshold", ticklabels(thresholds)),
        title="RT = $rt",
        cbar=false,
    )
    i, j = argmax(X).I
    annotate!(i, j, text("X", pointsize=8))
end
plot(figs..., size=(900,600))
savefig("figs/heatmaps")

# %% ==================== Posterior over drift, integrating out threshold ====================

figs = map(rts) do rt
    trial = (rt, 1)
    pdrift = map(vs) do drift
        quadgk(1e-3, 20) do threshold
            posterior(trial, drift, threshold)
        end |> first
    end
    plot(vs, pdrift, title="RT = $rt")
    vline!([vs[argmax(pdrift)]])
end
plot(figs..., size=(900,600))
savefig("figs/marginal")
