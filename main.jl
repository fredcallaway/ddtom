using Distributed
using Serialization
using JSON
using Dates
using Glob
using ProgressMeter

@everywhere include("model.jl")
include("figure.jl")
data = JSON.parsefile("results/trends_to_fit.json")

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

function compute_exp2_preds()
    stamp = Dates.format(now(), "Y-mm-dd-HH-MM-SS")
    name = "tmp/exp2-grid"
    mkpath(name)
    out = "$name/$stamp.jls"

    results = @showprogress pmap(grid) do (σ, θ)
        exp2_predictions(σ, θ)
    end

    serialize(out, (grid, results))
    println("Wrote ", out)
    return grid, results
end

exp2_files = sort!(glob("tmp/exp2-grid/*"))
USE_PRECOMPUTED = false
if USE_PRECOMPUTED && !isempty(exp2_files)
    println("Using precomputed exp2 predictions")
    grid2, exp2_results = deserialize(exp2_files[end])
else
    println("Computing exp2 predictions. This could take a while.")
    grid2, exp2_results = compute_exp2_preds()
end

@assert collect(grid2) == grid

exp2_keys = ["AA", "BC", "BA", "AC"]
exp2_targets = [data["Expt_2"][x] for x in exp2_keys]

rescale(r) = [1 - r[:AA], r[:BC], r[:BA], 1-r[:AC]] .* 100
exp2_predict(σ, θ) = rescale(exp2_results[findfirst(isequal((σ, θ)), grid)])
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

write("results/fitted_predictions.json", JSON.json(predictions))

# %% ==================== Predictions with default values ====================

σ, θ = .3, 3 
α = 100
θlo, θhi = 2.5, 3.5

predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(σ, θ, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(σ, θ)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(σ, θlo, θhi, α))
)

write("results/default_predictions.json", JSON.json(predictions))


# %% ==================== Sensitivity analysis ====================

# Find reasonable values: those for which the observed data is not
# highly improbable

@everywhere using SplitApplyCombine

@everywhere function data_plausible(σ, θ; plausible=1e-4)
    p1 = hquadrature(-6σ, 6σ) do v
        posterior((3., 1), v, θ, σ)
    end |> first
    p1 > plausible || return false

    p2 = hquadrature(-6σ, 6σ) do v
        posterior((9., 1), v, θ, σ)
    end |> first
    return p2 > plausible
end

@everywhere function reasonable_accuracy(σ, θ; lo=0.55, hi=0.95, N=10000)
    rt, choice = map(randn(N)) do x
       sample_choice_rt(abs(x) * σ, θ)
    end |> invert
    accuracy = mean(choice)
    0.55 < accuracy < 0.95
    
end

big_σs = 10 .^ (-2:.1:1)
# big_θs = (0.1:0.5:20)
big_θs = 10 .^ (-1:0.1:2)
big_grid = collect(Iterators.product(big_σs, big_θs))

plaus = @showprogress pmap(big_grid) do (σ, θ)
    data_plausible(σ, θ)
end

acc = @showprogress pmap(big_grid) do (σ, θ)
    reasonable_accuracy(σ, θ)
end

figure("plausibility") do
    heatmap(acc .& plaus)
end

reasonable = big_grid[acc .& plaus]

# %% --------
# Compute predictions

Expt_1 = map(reasonable) do (σ, θ)
    prediction = Dict(exp1_keys .=> exp1_predict(σ, θ, α))
    (;σ, θ, prediction)
end;

println("Computing expt 2 predictions. Might take a while...")
Expt_2 = @showprogress pmap(reasonable) do (σ, θ)
    prediction = exp2_predictions(σ, θ)
    (;σ, θ, prediction)
end;

grid3 = collect(Iterators.product(big_σs, big_θs, big_θs))[:]
filter!(grid3) do (σ, θlo, θhi)
    θhi > θlo && (σ, θlo) in reasonable && (σ, θhi) in reasonable
end

Exp_3 = map(grid3) do (σ, θlo, θhi)
    prediction = Dict(exp3_keys .=> exp3_predict(σ, θlo, θhi, α))
    (;σ, θ, prediction)
end

predictions = (;Expt_1, Expt_2, Exp_3)

write("results/sensitivity_analysis.json", JSON.json(predictions))

# %% --------

# The inferred value differenc monotonically decreasing with RT
@assert map(Expt_1) do res
    issorted([res.prediction[k] for k in exp1_keys]; rev=true)
end |> all

# %% --------

using SplitApplyCombine
R = map(Expt_2) do x                                                                                                  
   (;x.prediction...)                                                                                                                 
end |> invert      

# How often is each choice in the predicted direction?
@assert all(R.AA .< 0.5)
@show mean(R.BC .> 0.5)  # This one is weird, see below:
@assert all(R.BA .> 0.5)
@assert all(R.AC .< 0.5)

# %% --------
bad = findall(R.BC .< 0.5)

for i in bad
    # due to numerical error, these don't sum to one which is why p1 is less than 0.5
    p1 = predict_bc_choice((3., 2), (9., 2); σ, θ)
    p2 = predict_bc_choice((3., 2), (9., 2); σ, θ, choice=2)
    @assert p1 > p2
end

# %% --------
# All the differences and interaction are in the predicted direction
@assert map(Exp_3) do res
    x = res.prediction
    x["thLo9"] < x["thHi9"] &&
    x["thLo3"] < x["thHi3"] &&
    x["thLo9"] < x["thLo3"] &&
    x["thHi9"] < x["thHi3"] &&
    x["thHi3"] - x["thHi9"] > x["thLo3"] - x["thLo9"]
end |> all
