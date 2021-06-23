using ProgressMeter
using SplitApplyCombine
using Distributed

@everywhere begin
    include("fitting.jl")
    include("lba_model.jl")
    check_reasonable(model::LBA) = check_reasonable_base(model) && low_nochoice_rate(model)
end

# %% ==================== Sobol ====================

box = Box(
    β = (.01, 10, :log),
    β0 = (.01, 10, :log),
    θ = (.1, 100, :log),
    A = (.1, 100, :log),
    sv = 1,
)
models = sample_models(LBA, box)
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

@assert model isa LBA  # make sure I'm in the right terminal
write("results/lba_fitted_predictions.json", JSON.json(predictions))

using Serialization
serialize("fits/lba", model)
