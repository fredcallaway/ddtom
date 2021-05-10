using JSON

include("model.jl")
include("experiment2.jl")
include("box.jl")

data = JSON.parsefile("results/trends_to_fit.json")

sse(x, y) = sum((x .- y) .^ 2)

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

# %% ==================== Reasonable parameter check ====================

function data_plausible(model; plausible=1e-4, abstol=plausible/10, maxevals=10000)
    p1, ε = hquadrature(-6, 6; abstol, maxevals) do pref
        posterior(model, Observation(true, 3.), pref)
    end
    (ε < abstol && p1 > plausible) || return false

    p2, ε = hquadrature(-6, 6; abstol, maxevals) do pref
        posterior(model, Observation(true, 9.), pref)
    end
    
    return (ε < abstol && p2 > plausible)
end

function reasonable_accuracy(model; lo=0.55, hi=0.95, N=1000)
    accuracy = map(randn(N)) do x
       simulate(model, abs(x)).choice == 1
    end |> mean
    lo < accuracy < hi
end

function low_nochoice_rate(model; max_rate=0.05 , N=1000)
    nochoice_rate = map(randn(N)) do x
       simulate(model, abs(x)).choice == 0
    end |> mean
    nochoice_rate < max_rate
end


# %% ==================== Experiment 1 ====================

exp1_rts = [3,5,7,9]
exp1_keys = ["$(t)sec" for t in exp1_rts]
exp1_targets = [data["Expt_1"][k] for k in exp1_keys]
exp1_predict(model::Model, α) = α .* posterior_mean_pref.([model], exp1_rts)
exp1_loss(model::Model, α=optimize_α(model)) = sse(exp1_predict(model, α), exp1_targets)

function optimize_α(model, bound=1000)
    α = optimize(0, 1000) do α
        sse(exp1_predict(model, α), exp1_targets)
    end |> Optim.minimizer
    if α > 0.95 * bound
        if bound > 1e8
            @warn "hit max bound"
            return α
        end
        return optimize_α(model, bound*10)
    end
    α
end


# %% ==================== Experiment 2 ====================

rescale(r) = [1 - r[:AA], r[:BC], r[:BA], 1-r[:AC]] .* 100
exp2_keys = ["AA", "BC", "BA", "AC"]
exp2_targets = [data["Expt_2"][x] for x in exp2_keys]

exp2_predict(model::Model) = rescale(exp2_predictions(model))
exp2_loss(model::Model) = sse(exp2_predict(model), exp2_targets)


# %% ==================== Experiment 3 ====================

exp3_keys = ["thHi3", "thLo3", "thHi9", "thLo9"]
exp3_targets = [data["Expt_3"][k] for k in exp3_keys]

function exp3_predict(model, θlo, θhi, α)
    args = [(3, θhi), (3, θlo), (9, θhi), (9, θlo)]
    [2α * posterior_mean_pref(mutate(model; θ), rt) for (rt, θ) in args]
end

exp3_loss(model, θlo, θhi, α) = sse(exp3_predict(model, θlo, θhi, α), exp3_targets)

function optimize_θs(model, α)
    res = optimize([model.θ, model.θ]) do (θlo, θhi)
        exp3_loss(model, θlo, θhi, α)
    end
    res.minimizer
end


