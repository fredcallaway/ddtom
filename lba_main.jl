@everywhere using Revise
using JSON
using Sobol
using ProgressMeter
using SplitApplyCombine
using Printf

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
    lo < accuracy < hi
end

@everywhere function low_nochoice_rate(model; max_rate=0.05 , N=1000)
    nochoice_rate = map(randn(N)) do x
       simulate(model, abs(x)).choice == 0
    end |> mean
    nochoice_rate < max_rate
end

# %% --------
box = Box(
    β = (0, 10),
    β0 = (0, 10),
    θ = (0, 100),
    A = (0, 100),
    sv = 1,
)
# %% --------
N = 10000
xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect
is_reasonable = @showprogress map(xs) do x
    model = LBA(;box(x)...)
    model.A < model.θ
    data_plausible(model) && reasonable_accuracy(model) && low_nochoice_rate(model)
end
@show mean(is_reasonable)

models = map(xs[is_reasonable]) do x
    LBA(;box(x)...)
end
@show length(models)

println("Reasonable ranges")
foreach(keys(box.dims), invert(xs[is_reasonable])) do k, x
    x = rescale(box[k], x)
    @printf "%2s   %1.3f %1.3f\n" k quantile(x, .01) quantile(x, .99)
end

cor(combinedims(xs[is_reasonable])')

# %% ==================== Experiment 1 ====================
@everywhere begin
    exp1_rts = [3,5,7,9]
    exp1_keys = ["$(t)sec" for t in exp1_rts]
    exp1_targets = [data["Expt_1"][k] for k in exp1_keys]
    exp1_predict(model::Model, α) = α .* posterior_mean_pref.([model], exp1_rts)
end

Expt_1 = @showprogress pmap(models) do model
    α = optimize(0, 500) do α
        sse(exp1_predict(model, α), exp1_targets)
    end |> Optim.minimizer
    prediction = Dict(exp1_keys .=> exp1_predict(model, α))
    (;prediction)
end;

pass_expt1 = map(Expt_1) do res
    issorted([res.prediction[k] for k in exp1_keys]; rev=true)
end
weird_models = models[.!pass_expt1]
# %% --------
map(Expt_1[.!pass_expt1]) do res
    x = [res.prediction[k] for k in exp1_keys]
    x
end

# %% --------
model = weird_models[3]

x = 3:.1:10
y = @showprogress map(x) do rt
    posterior_mean_pref(model, rt)
end
figure() do
    plot(x, y)
end
# %% --------
x = 3:.1:10
figure() do
    foreach(0:0.2:1) do vd
        y = @showprogress map(x) do rt
            likelihood(model, Observation(true, rt), vd)
        end
        plot!(x, y)
    end
end

# %% --------

x = -3:.1:3
map(x) do vd
    likelihood(model, Observation(true, vd))




# %% ==================== Experiment 2 ====================

choose_first = @showprogress pmap(models) do model
    exp2_predictions(model; choice=true)
end

choose_second = @showprogress pmap(models) do model
    exp2_predictions(model; choice=false)
end

# %% --------

rescale(r) = [1 - r[:AA], r[:BC], r[:BA], 1-r[:AC]] .* 100
exp2_keys = ["AA", "BC", "BA", "AC"]
exp2_targets = [data["Expt_2"][x] for x in exp2_keys]

exp2_loss = map(e2) do d
    sse(rescale(d), exp2_targets)
end

# findmin(exp2_loss)


# %% --------
using SplitApplyCombine
R = invert([(;e...) for e in e2])

# How often is each choice in the predicted direction?
@show mean(R.AA .< 0.5)
@show mean(R.BC .> 0.5)  # This one is weird, see below:
@show mean(R.BA .> 0.5)
@show mean(R.AC .< 0.5)


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


# %% ==================== GP minimize ====================
exp2_predict(model) = rescale(exp2_predictions(model))

x = exp2_predict(model)

include("gp_min.jl")
result_gp = gp_minimize(length(box); iterations=1000, verbose=true) do x
    model = LBA(;box(x)...)
    e1_loss = optimize(0, 500) do α
        sse(exp1_predict(model, α), exp1_targets)
    end |> Optim.minimum

    e2_loss = sse(exp2_predict(model), exp2_targets)

    √(e1_loss + e2_loss) / 10
end



