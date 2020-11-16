using Distributed
using Serialization
using JSON
using Dates
using Glob
using ProgressMeter

@everywhere include("model.jl")
include("figure.jl")
data = JSON.parsefile("trends_to_fit.json")

σs = 0.1:0.1:1
θs = (0.2:0.2:3)
grid = collect(Iterators.product(σs, θs))
sse(x, y) = sum((x .- y) .^ 2)


# %% ==================== Experiment 1 ====================
exp1_rts = [3,5,7,9]
exp1_keys = ["$(t)sec" for t in exp1_rts]
exp1_targets = [data["Expt_1"][k] for k in exp1_keys]

exp1_predict(σ, θ, α) = α .* posterior_mean_drift.(exp1_rts, θ, σ)

function exp1_loss(σ, θ)
    res = optimize(0, 500) do α
        sse(exp1_predict(σ, θ, α), exp1_targets)
    end
    (α=res.minimizer, loss=res.minimum)
end

L1 = map(grid) do (σ, θ)
    exp1_loss(σ, θ).loss
end

figure("L1") do
    X = copy(L1)
    X[X .> 50] .= NaN
    heatmap(X)
end

# %% ==================== Experiment 2 ====================
@everywhere include("experiment2.jl")

function precompute_exp2_preds()
    stamp = Dates.format(now(), "mm-dd-HH-MM-SS")
    name = "tmp/exp2-grid"
    mkpath(name)
    out = "$name/$stamp.jls"

    results = @showprogress pmap(grid) do (σ, θ)
        exp2_predictions(σ, θ)
    end

    serialize(out, (grid, results))
    println("Wrote ", out)
end


# load the most recent (this year!) precomputed grid
grid_files = sort!(glob("tmp/exp2-grid/*")
if isempty(grid_files)
    println("Warning: precomputing experiment 2 predictions. Hopefully you have multiple cpus.")
    precompute_exp2_preds()
end
grid2, results = deserialize(grid_files)[end])
@assert collect(grid2) == grid


exp2_keys = ["AA", "BC", "BA", "AC"]
exp2_targets = [data["Expt_2"][x] for x in exp2_keys]

rescale(r) = [1 - r[:AA], r[:BC], r[:BA], 1-r[:AC]] .* 100
exp2_predict(σ, θ) = rescale(results[findfirst(isequal((σ, θ)), grid)])
exp2_loss(σ, θ) = sse(exp2_predict(σ, θ), exp2_targets)

L2 = map(grid) do (σ, θ)
    exp2_loss(σ, θ)
end

figure("L2") do
    heatmap(L2)
end

# %% ==================== Best fit across 1 and 2 ====================

σ, θ = grid[argmin(L1 .+ L2)]
α = exp1_loss(σ, θ).α


# %% ==================== Experiment 3 ====================

exp3_keys = ["thHi3", "thLo3", "thHi9", "thLo9"]
exp3_targets = [data["Expt_3"][k] for k in exp3_keys]

function exp3_predict(σ, θlo, θhi, α)
    args = [(3, θhi), (3, θlo), (9, θhi), (9, θlo)]
    [2α * posterior_mean_drift(a..., σ) for a in args]
end

exp3_loss(σ, θlo, θhi, α) = sse(exp3_predict(σ, θlo, θhi, α), exp3_targets)

res = optimize([θ, θ]) do (θlo, θhi)
    exp3_loss(σ, θlo, θhi, α)
end
θlo, θhi = res.minimizer

# %% ==================== Best fit across all ====================

# function total_loss(σ, θ)
#     res = optimize([θ, θ, 100.]) do (θlo, θhi, α)
#         sse(exp3_predict(σ, θlo, θhi, α), exp3_targets) +
#         sse(exp1_predict(σ, θ, α), exp1_targets)
#     end
#     (loss = sse(exp2_predict(σ, θ), exp2_targets) + res.minimum,
#      params=res.minimizer)
# end

# using ProgressMeter
# TL = @showprogress map(grid) do (σ,θ)
#     total_loss(σ,θ)
# end
# best = argmin(first.(TL))

# mle = (
#     σ = grid[best][1],
#     θ = grid[best][2],
#     θlo = TL[best].params[1],
#     θhi = TL[best].params[2],
#     α = TL[best].params[3],
# )


# %% ==================== Save predictions ====================

@show σ θ θlo θhi α
predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(σ, θ, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(σ, θ)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(σ, θlo, θhi, α))
)

write("fitted_predictions.json", JSON.json(predictions))


# %% ==================== Predictions with default values ====================

σ, θ = .3, 3 
α = 100
θlo, θhi = 2.5, 3.5

predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(σ, θ, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(σ, θ)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(σ, θlo, θhi, α))
)

write("default_predictions.json", JSON.json(predictions))

