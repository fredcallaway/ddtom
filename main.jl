using Distributed
using Serialization
using JSON

@everywhere include("model.jl")
data = JSON.parsefile("trends_to_fit.json")

σs = 0.1:0.1:1
θs = (0.2:0.2:3)
grid = collect(Iterators.product(σs, θs))
sse(x, y) = sum((x .- y) .^ 2)

# %% ==================== Choose parameters ====================
σ = 0.3
θ = 2.2

rts = 0:0.02:10
foo(v, θ) = [likelihood((rt, 1), v, θ) for rt in rts]

figure() do
    for v in 0:0.5:3
        plot!(rts, foo(v, θ); label="$v")
    end
end

quadgk(0, 50) do rt
    likelihood((rt, 1), 2.5, θ)
end |> first

# %% --------

function predict_choice_rt(v, θ)
    choice = quadgk(0, 50) do rt
        likelihood((rt, 1), v, θ)
    end |> first
    rt = quadgk(0, 50) do rt
        p0 = likelihood((rt, 0), v, θ)
        p1 = likelihood((rt, 1), v, θ)
        (p0 + p1) * rt
    end |> first
    (choice, rt)
end
using SplitApplyCombine

# predict_choice_rt(0., θ)
# vs = abs.(rand(Normal(0, √2 * σ), 500))
# @time preds = predict_choice_rt.(vs, θ);
# %% --------

xs = 0:0.1:3

options = [
    (0.3, 3),
    (0.3, 3.5),
    (0.3, 2.5)
    # (0.3, 3)
    # (0.4, 3),
    # (0.5, 3),
]



figure() do 
    choice_plot = plot(ylim=(0.5,1), legend=:bottomright, 
        xlabel="Value Difference", ylabel="P(Choose preferred item)")
    rt_plot = plot(ylim=(0,9), 
        xlabel="Value Difference", ylabel="RT (seconds)")
    foreach(options) do (σ, θ)
        choice, rt = map(xs) do x
            predict_choice_rt(σ * x, θ)
        end |> invert
        plot!(choice_plot, xs, choice, label="(σ=$σ, θ=$θ)")
        plot!(rt_plot, xs, rt)
    end
    qs = quantile.(truncated(Normal(0, √2), 0, Inf), range(0,1;length=5)[1:end-1])
    for p in (choice_plot, rt_plot)
        vline!(p, qs, color=:gray, alpha=0.4)
    end
    plot(choice_plot, rt_plot, size=(600,300))
end

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
@everywhere include("transitive.jl")
using Serialization
using Dates
using Glob

# %% -------- Precompute a grid
stamp = Dates.format(now(), "mm-dd-HH-MM-SS")
name = "tmp/exp2-grid"
mkpath(name)
out = "$name/$stamp.jls"

results = pmap(grid) do (σ, θ)
    @time exp2_predictions(σ, θ)
end

serialize(out, (grid, results))
println("Wrote ", out)

# %% --------
exp2_keys = ["AA", "BC", "BA", "AC"]
exp2_targets = [data["Expt_2"][x] for x in exp2_keys]

grid2, results = deserialize(sort!(glob("tmp/exp2-grid/*"))[end])
@assert collect(grid2) == grid
rescale(r) = [1 - r[:AA], r[:BC], r[:BA], 1-r[:AC]] .* 100
exp2_predict(σ, θ) = rescale(results[findfirst(isequal((σ, θ)), grid)])
exp2_loss(σ, θ) = sse(exp2_predict(σ, θ), exp2_targets)

L2 = map(grid) do (σ, θ)
    exp2_loss(σ, θ)
end

figure("L2") do
    heatmap(L2)
end

exp2_predict(0.3, 3)

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

function total_loss(σ, θ)
    res = optimize([θ, θ, 100.]) do (θlo, θhi, α)
        sse(exp3_predict(σ, θlo, θhi, α), exp3_targets) +
        sse(exp1_predict(σ, θ, α), exp1_targets)
    end
    (loss = sse(exp2_predict(σ, θ), exp2_targets) + res.minimum,
     params=res.minimizer)
end

using ProgressMeter
TL = @showprogress map(grid) do (σ,θ)
    total_loss(σ,θ)
end
best = argmin(first.(TL))

mle = (
    σ = grid[best][1],
    θ = grid[best][2],
    θlo = TL[best].params[1],
    θhi = TL[best].params[2],
    α = TL[best].params[3],
)


# %% ==================== Save predictions ====================
σ, θ = .3, 3 
α = 100
θlo, θhi = 2.5, 3.5

predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(σ, θ, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(σ, θ)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(σ, θlo, θhi, α))
)

write("default_predictions.json", JSON.json(predictions))


σ, θ = grid[best]
θlo, θhi, α = TL[best].params
@show σ θ θlo θhi α
predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(σ, θ, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(σ, θ)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(σ, θlo, θhi, α))
)

write("fitted_predictions.json", JSON.json(predictions))




# %% --------




α .* posterior_mean_drift(3, θlo, σ)
α .* posterior_mean_drift(9, θhi, σ)
α .* posterior_mean_drift(9, θlo, σ)

exp3_predict(σ, θ1, θ2, α)


L1 = let
    rts = [3,9]
    # targets = [data["Expt_1"]["$(t)sec"] for t in ts]
    map(grid) do (σ, θ)
        res = optimize(0, 500) do α
            pred = α .* posterior_mean_drift.(exp1.rt, θ, σ)
            sum((pred .- exp1.values_mean) .^ 2)
        end
        res.minimum
    end
end

figure("L1") do
    X = copy(L1)
    X[X .> 50] .= NaN
    heatmap(X)
end






# %% ==================== MLE for exp2 ====================


bounds = ([-3, -3, -3, 0.], [3, 3, 3, 20.])  # x, y, z, θ

trials = define_trials()[:XX]
res = optimize(bounds..., rand(4), SAMIN(), Optim.Options(iterations=10^4)) do args
    -xyxz_likelihood(trials..., args...)
end
@show -res.minimum round.(res.minimizer; digits=3)
# %% --------

xyxz_likelihood(trials..., res.minimizer...)
xy_trial, xz_trial = trials
x, y, z, θ = res.minimizer
likelihood(xy_trial, x - y, θ)

θ

likelihood(xz_trial, x - z, θ)

# %% ==================== MAP for exp2 ====================


res = optimize(bounds..., rand(4), SAMIN(), Optim.Options(iterations=10^5)) do vv
    -log(xyzθ_prior(vv...; σ=3, λ=20) * xyxz_likelihood(trials..., vv...))
end;
@show -res.minimum round.(res.minimizer; digits=3)
x, y, z, θ = res.minimizer
println(round.(res.minimizer; digits=3), " => ", round(res.minimum; sigdigits=3))



# %% --------




# %% --------
include("model.jl")

using Cubature
using BenchmarkTools

# %% --------

function gen_trials(drift, θ; N=10)
    dd = ConstDrift(drift, dt)
    bb = ConstSymBounds(θ, dt)
    smp = sampler(dd, bb)
    [rand(smp) for i in 1:N]
end

function p_upper(drift, θ; max_rt=50)
    dd = ConstDrift(drift, dt)
    bb = ConstSymBounds(θ, dt)
    quadgk(0, max_rt; atol=1e-5) do x  # integration
        pdfu(dd, bb, x)
    end |> first
end

@btime p_upper(1., 1.)
trials = gen_trials(0.2, 1; N=100000)
choice = map(x->x[2], trials)
@assert ≈(mean(choice), p_upper(0.2, 1); atol=.01)


# %% --------

v = (x=3, y=2, z=1)
θ = 1
gen_trials(v.x - v.y, θ)


# Question type XX
inf1 = MAP_drift((fastrt, 1), θ) # choose between X (1) and Y (?), choose X
inf2 = MAP_drift((slowrt, 1), θ) # choose between X (1) and Z (?), choose X
print("$inf1-$inf2") # need value difference btw Y and Z

# Question type xx
inf1 = MAP_drift((fastrt, ?), θ) # choose between X (1) and Y (?), choose Y
inf2 = MAP_drift((slowrt, ?), θ) # choose between X (1) and Z (?), choose Z
print("$inf1-$inf2") # need value difference btw Y and Z

# Question type Xxfast
inf1 = MAP_drift((slowrt, 1), θ) # choose between X (1) and Y (?), choose X
inf2 = MAP_drift((fastrt, ?), θ) # choose between X (1) and Z (?), choose Z
print("$inf1-$inf2") # need value difference btw Y and Z

# Question type Xxslow
inf1 = MAP_drift((fastrt, 1), θ) # choose between X (1) and Y (?), choose X
inf2 = MAP_drift((slowrt, ?), θ) # choose between X (1) and Z (?), choose Z

# %% --------



@time predict_yz_choice(1,1,2.,1.)



