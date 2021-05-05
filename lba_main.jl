@everywhere using Revise
using JSON
using Sobol
using ProgressMeter

@everywhere begin
    includet("model.jl")
    includet("lba_base.jl")
    includet("lba_model.jl")
    includet("experiment2.jl")
    includet("box.jl")
end

data = JSON.parsefile("results/trends_to_fit.json")
@everywhere data = $data
@everywhere sse(x, y) = sum((x .- y) .^ 2)

# %% ==================== Find parameters ====================

@everywhere function data_plausible(model; plausible=1e-4, abstol=plausible/10, maxevals=10000)
    p1, ε = hquadrature(-6, 6; abstol, maxevals) do pref
        posterior(model, Observation(true, 3.), pref)
    end
    (ε < abstol && p1 > plausible) || return false

    p2, ε = hquadrature(-6, 6; abstol, maxevals) do pref
        posterior(model, Observation(true, 9.), pref)
    end
    
    return (ε < abstol && p2 > plausible)
end


@everywhere function reasonable_accuracy(model; lo=0.55, hi=0.95, N=1000)
    accuracy = map(randn(N)) do x
       simulate(model, abs(x)).choice == 1
    end |> mean
    0.55 < accuracy < 0.95    
end

# %% --------

N = 100000
@everywhere box = Box(
    β = (0, 1),
    β0 = (0, 1),
    θ = (0, 2),
    A = (0, 2),
    sv = (0, 1),
)
xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect
is_reasonable = @showprogress pmap(xs) do x
    model = LBA(;box(x)...)
    data_plausible(model) && reasonable_accuracy(model)
end
models = map(xs[is_reasonable]) do x
    LBA(;box(x)...)
end

# %% ==================== Experiment 1 ====================
exp1_rts = [3,5,7,9]
exp1_keys = ["$(t)sec" for t in exp1_rts]
exp1_targets = [data["Expt_1"][k] for k in exp1_keys]

exp1_predict(model::Model, α) = α .* posterior_mean_pref.([model], exp1_rts)

e1 = exp1_predict(model, 100.)
@assert issorted(e1; rev=true)

# %% ==================== Experiment 2 ====================

e2 = (;exp2_predictions(model)...)

# How often is each choice in the predicted direction?
@assert e2.AA .< 0.5
@assert e2.BC .> 0.5
@assert e2.BA .> 0.5
@assert e2.AC .< 0.5

# %% ==================== Experiment 3 ====================

exp3_keys = ["thHi3", "thLo3", "thHi9", "thLo9"]

function mutate(x::T; kws...) where T
    for field in keys(kws)
        if !(field in fieldnames(T))
            error("$(T.name) has no field $field")
        end
    end
    return T([get(kws, fn, getfield(x, fn)) for fn in fieldnames(T)]...)
end

function exp3_predict(θlo, θhi; α=100)
    args = [(3, θhi), (3, θlo), (9, θhi), (9, θlo)]
    [2α * posterior_mean_pref(mutate(model; θ), rt) for (rt, θ) in args]
end

e3 = Dict(exp3_keys .=> exp3_predict(1.9, 2.1))

@assert e3["thLo9"] < e3["thHi9"]
@assert e3["thLo3"] < e3["thHi3"]
@assert e3["thLo9"] < e3["thLo3"]
@assert e3["thHi9"] < e3["thHi3"]
@assert e3["thHi3"] - e3["thHi9"] > e3["thLo3"] - e3["thLo9"]
