using Distributed
using Serialization
using JSON
using Dates
using Glob
using ProgressMeter

@everywhere include("model.jl")
@everywhere include("experiment2.jl")

include("figure.jl")

data = JSON.parsefile("results/trends_to_fit.json")
USE_PRECOMPUTED = false
DISABLE_PLOTTING = true
βs = 0.1:0.1:1
θs = (0.2:0.2:3)
grid = collect(Iterators.product(βs, θs))
sse(x, y) = sum((x .- y) .^ 2)


# %% ==================== Experiment 1 ====================
exp1_rts = [3,5,7,9]
exp1_keys = ["$(t)sec" for t in exp1_rts]
exp1_targets = [data["Expt_1"][k] for k in exp1_keys]

exp1_predict(model::Model, α) = α .* posterior_mean_pref.([model], exp1_rts)

function exp1_loss(β, θ)
    model = DDM(θ, β)
    res = optimize(0, 500) do α
        sse(exp1_predict(model, α), exp1_targets)
    end
    (α=res.minimizer, loss=res.minimum)
end

L1 = map(grid) do (β, θ)
    exp1_loss(β, θ).loss
end

figure("L1") do
    X = copy(L1)
    X[X .> 50] .= NaN
    heatmap(X)
end


# %% ==================== Experiment 2 ====================

function compute_exp2_preds()
    stamp = Dates.format(now(), "Y-mm-dd-HH-MM-SS")
    name = "tmp/exp2-grid"
    mkpath(name)
    out = "$name/$stamp.jls"

    results = @showprogress pmap(grid) do (β, θ)
        exp2_predictions(DDM(θ, β))
    end

    serialize(out, (grid, results))
    println("Wrote ", out)
    return grid, results
end

exp2_files = sort!(glob("tmp/exp2-grid/*"))
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

@everywhere rescale(r) = [1 - r[:AA], r[:BC], r[:BA], 1-r[:AC]] .* 100
exp2_predict(model) = rescale(exp2_results[findfirst(isequal((model.β, model.θ)), grid)])
exp2_loss(β, θ) = sse(exp2_predict(DDM(θ, β)), exp2_targets)

L2 = map(grid) do (β, θ)
    exp2_loss(β, θ)
end

figure("L2") do
    heatmap(L2)
end

# %% ==================== Best fit across 1 and 2 ====================

β, θ = grid[argmin(L1 .+ L2)]
α = exp1_loss(β, θ).α

@info "Best fitting parameters" β θ α



# %% ==================== Experiment 3 ====================

exp3_keys = ["thHi3", "thLo3", "thHi9", "thLo9"]
exp3_targets = [data["Expt_3"][k] for k in exp3_keys]

function exp3_predict(β, θlo, θhi, α)
    args = [(3, θhi), (3, θlo), (9, θhi), (9, θlo)]
    [2α * posterior_mean_pref(DDM(θ, β), rt) for (rt, θ) in args]
end

exp3_loss(β, θlo, θhi, α) = sse(exp3_predict(β, θlo, θhi, α), exp3_targets)

res = optimize([θ, θ]) do (θlo, θhi)
    exp3_loss(β, θlo, θhi, α)
end
θlo, θhi = res.minimizer

# %% ==================== Best fit across all ====================

# function total_loss(β, θ)
#     res = optimize([θ, θ, 100.]) do (θlo, θhi, α)
#         sse(exp3_predict(β, θlo, θhi, α), exp3_targets) +
#         sse(exp1_predict(β, θ, α), exp1_targets)
#     end
#     (loss = sse(exp2_predict(β, θ), exp2_targets) + res.minimum,
#      params=res.minimizer)
# end

# using ProgressMeter
# TL = @showprogress map(grid) do (β,θ)
#     total_loss(β,θ)
# end
# best = argmin(first.(TL))

# mle = (
#     β = grid[best][1],
#     θ = grid[best][2],
#     θlo = TL[best].params[1],
#     θhi = TL[best].params[2],
#     α = TL[best].params[3],
# )


# %% ==================== Save predictions ====================

@show β θ θlo θhi α
fit_model = DDM(θ, β)
predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(fit_model, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(fit_model)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(β, θlo, θhi, α))
)

write("results/fitted_predictions.json", JSON.json(predictions))


# %% ==================== Sensitivity analysis ====================

# Find reasonable values: those for which the observed data is not
# highly improbable

@everywhere using SplitApplyCombine

@everywhere function data_plausible(β, θ; plausible=1e-4)
    p1 = hquadrature(-6, 6) do pref
        posterior(DDM(θ, β), Observation(true, 3.), pref)
    end |> first
    p1 > plausible || return false

    p2 = hquadrature(-6, 6) do pref
        posterior(DDM(θ, β), Observation(true, 9.), pref)
    end |> first
    return p2 > plausible
end

@everywhere function reasonable_accuracy(β, θ; lo=0.55, hi=0.95, N=10000)
    accuracy = map(randn(N)) do x
       simulate(DDM(θ, β), abs(x)).choice
    end |> mean
    0.55 < accuracy < 0.95    
end

big_βs = 10 .^ (-2:.1:1)
big_θs = 10 .^ (-1:0.1:2)
big_grid = collect(Iterators.product(big_βs, big_θs))

plaus = @showprogress pmap(big_grid) do (β, θ)
    data_plausible(β, θ)
end

acc = @showprogress pmap(big_grid) do (β, θ)
    reasonable_accuracy(β, θ)
end

reasonable = big_grid[acc .& plaus]

# %% --------
pyplot()

function make_heat(X, color)
    c = cgrad([color, "#ffffff"])
    X = float.(X)
    X[X .== 0] .= NaN
    heatmap!(X; c,
        yaxis=("β", (idx, big_βs[idx])), 
        xaxis=("θ", (idx, big_θs[idx])),
        cbar=false, grid=false, framestyle=:box
    )
end
figure("plausibility", dpi=500) do
    make_heat(plaus, "#81C0FF")
    make_heat(acc, "#FFE783")
    make_heat(acc .& plaus, "#7DE87D")
end

# %% --------
# Compute predictions



Expt_1 = map(reasonable) do (β, θ)
    α = exp1_loss(β, θ).α
    prediction = Dict(exp1_keys .=> exp1_predict(DDM(θ, β), α))
    (;β, θ, α, prediction)
end;

println("Computing expt 2 predictions. Might take a while...")
Expt_2 = @showprogress pmap(reasonable) do (β, θ)
    prediction = rescale(exp2_predictions(DDM(θ, β)))
    (;β, θ, prediction)
end;

grid3 = collect(Iterators.product(big_βs, big_θs, big_θs))[:]
filter!(grid3) do (β, θlo, θhi)
    θhi > θlo && (β, θlo) in reasonable && (β, θhi) in reasonable
end

Exp_3 = map(grid3) do (β, θlo, θhi)
    α = optimize(0, 500) do α
        exp3_loss(β, θlo, θhi, α)
    end |> Optim.minimizer
    prediction = Dict(exp3_keys .=> exp3_predict(β, θlo, θhi, α))
    (;β, θ, α, prediction)
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
    p1 = predict_bc_choice(DDM(θ, β), Observation(false, 3.), Observation(false, 9.))
    p2 = predict_bc_choice(DDM(θ, β), Observation(false, 3.), Observation(false, 9.); choice=false)
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
