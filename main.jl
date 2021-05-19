using ProgressMeter
using SplitApplyCombine
using Distributed

@everywhere include("fitting.jl")

# %% ==================== Sobol ====================
# βs = 0.1:0.1:1
# θs = (0.2:0.2:3)

@everywhere box = Box(
    β = (.01, 10, :log),
    θ = (0.1, 100, :log),
)
models = sample_models(DDM, box)
sobol_loss = @showprogress pmap(joint_loss, models)

# %% ==================== Nelder Mead ====================

y = map(scalar_loss, sobol_loss)
top30 = models[partialsortperm(y, 1:30)]
finetune_results = @showprogress pmap(top30) do model
    finetune(model, box)
end

# %% --------

models, loss = invert(finetune_results)
min_y, min_i = findmin(loss)
model = models[min_i]

# %% ==================== Save predictions ====================

α = optimize_α(model)
θlo, θhi = optimize_θs(model, α)

predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(model, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(model)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(model, θlo, θhi, α))
)

@assert model isa DDM  # make sure I'm in the right terminal heheh
write("results/ddm_fitted_predictions.json", JSON.json(predictions))

using Serialization
serialize("tmp/ddm_fit", model)