using Sobol
using ProgressMeter
using SplitApplyCombine
using Printf
using Random

@everywhere include("fitting.jl")
include("figure.jl")
DISABLE_PLOTTING = true
# pyplot()

# %% ==================== Identify reasonable paramaters ====================
Random.seed!(123)

big_βs = 10 .^ (-2:.1:1)
big_θs = 10 .^ (-1:0.1:2)
big_grid = collect(Iterators.product(big_βs, big_θs))

plaus = @showprogress map(big_grid) do (β, θ)
    data_plausible(DDM(;β, θ))
end

acc = @showprogress pmap(big_grid) do (β, θ)
    reasonable_accuracy(DDM(;β, θ); N=10^6)
end

reasonable = big_grid[acc .& plaus]
models = map(reasonable) do (β, θ)
    DDM(;β, θ)
end

function make_heat(X, color)
    c = cgrad([color, "#ffffff"])
    X = float.(X)
    X[X .== 0] .= NaN
    idx = 1:10:31
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

# %% ==================== Experiment 1 ====================

Expt_1 = map(models) do model
    α = optimize_α(model)
    prediction = Dict(exp1_keys .=> exp1_predict(model, α))
    (;model.β, model.θ, α, prediction)
end;

@assert map(Expt_1) do res
    issorted([res.prediction[k] for k in exp1_keys]; rev=true)
end |> all

# %% ==================== Experiment 2 ====================

Expt_2 = @showprogress pmap(models) do model
    prediction = Dict(exp2_keys .=> rescale(exp2_predictions(model)))
    (;model.β, model.θ, prediction)
end;

R = map(Expt_2) do x
   (;Dict(Symbol(k) => v for (k, v) in pairs(x.prediction))...)
end |> invert

#= 
The integration is not super robust for some of the more extreme parameter
values. We try to correct for this by computing the probability of the 
inverse choice and taking the predicted probability to be 
=#

R2_uninv = @showprogress pmap(models) do model
    prediction = Dict(exp2_keys .=> rescale(exp2_predictions(model; choice=false)))
   (;Dict(Symbol(k) => v for (k, v) in pairs(prediction))...)
end 
R2 = invert(R2_uninv)

# %% --------




# %% --------
alt_Expt_2 = @showprogress pmap(models) do model
    prediction = Dict(exp2_keys .=> rescale(exp2_predictions(model; relative=true)))
    (;model.β, model.θ, prediction)
end;

alt_R = map(Expt_2) do x
   (;Dict(Symbol(k) => v for (k, v) in pairs(x.prediction))...)
end |> invert

#= 
The integration is not super robust for some of the more extreme parameter
values. We try to correct for this by computing the probability of the 
inverse choice and taking the predicted probability to be 
=#

alt_R2_uninv = @showprogress pmap(models) do model
    prediction = Dict(exp2_keys .=> rescale(exp2_predictions(model; relative=true, choice=false)))
   (;Dict(Symbol(k) => v for (k, v) in pairs(prediction))...)
end 
alt_R2 = invert(R2_uninv)



# %% ==================== More heatmaps ====================

X = map(fieldnames(typeof(R))) do k
    95 .< getfield(R, k) .+ getfield(R2, k) .< 105
end |> collect |> combinedims
good = all(X; dims=2)[:]



mask = acc .& plaus
reasonable_indices = findall(mask)
bad_integral_indices = reasonable_indices[.!good]
bad_integral = zeros(Int, size(mask))
bad_integral[bad_integral_indices] .= 1


figure("plausibility2", dpi=250) do
    make_heat(plaus, "#81C0FF")
    make_heat(acc, "#FFE783")
    make_heat(acc .& plaus, "#7DE87D")
    make_heat(bad_integral, "#668866")
end

# %% --------
@everywhere function data_prob(model, ab_trial, bc_trial)
    function score((a, b, c))
        abc_prior(a, b, c) * abc_likelihood(model, ab_trial, bc_trial, a, b, c)
    end

    # estimate the normalizing constant
    Z, ε = hcubature(score, make_bounds()..., maxevals=10^10)
    if ε > 1e-8  # this one has to be more accurate to ensure predict_bc_choice < 1
        # @error "abc_posterior: integral did not converge" model ab_trial bc_trial
        return NaN
    end
    Z
end

@everywhere function data_prob(model)
    fast, slow = 3., 9.
    data = [
        (Observation(true, fast), Observation(true, slow)),
        (Observation(false, fast), Observation(false, slow)),
        (Observation(false, fast), Observation(true, slow)),
        (Observation(true, fast), Observation(false, slow)),
    ]
    sum([log(data_prob(model, x, y)) for (x, y) in data])
end

data_logp = @showprogress pmap(data_prob, models)

figure() do
    # make_heat(plaus, "#81C0FF")
    reasonable_indices = findall(mask)
    X = fill(NaN, size(mask))
    X[reasonable_indices] .= data_logp
    idx = 1:10:31
    heatmap!(X;
        yaxis=("β", (idx, big_βs[idx])), 
        xaxis=("θ", (idx, big_θs[idx])),
        grid=false, framestyle=:box, title="log p(data)"
    )
end

# %% --------

function accuracy(model; N=10000)
    accuracy = map(randn(N)) do x
       simulate(model, abs(x)).choice == 1
    end |> mean
end

raw_acc = @showprogress map(accuracy, models)

figure() do
    reasonable_indices = findall(mask)
    X = fill(NaN, size(mask))
    X[reasonable_indices] .= raw_acc
    idx = 1:10:31
    heatmap!(X;
        yaxis=("β", (idx, big_βs[idx])), 
        xaxis=("θ", (idx, big_θs[idx])),
        grid=false, framestyle=:box, title="choice accuracy"
    )
end

# %% --------

X = map(fieldnames(typeof(R))) do k
    getfield(R, k) .- getfield(R2, k)
end |> collect |> combinedims

X = map(fieldnames(typeof(R))) do k
    getfield(R, k)
end |> collect |> combinedims

X = map(fieldnames(typeof(R))) do k
    getfield(R, k) .+ getfield(R2, k)
end |> collect |> combinedims






# %% ==================== Old ====================


β, θ = map(models[good[:]]) do x
    (;x.β, x.θ)
end |> invert
maximum(θ)

# %% --------
bad = models[findall(R.AC .+ R2.AC .> 101)]
plausible=1e-4; abstol=plausible/100; maxevals=100000
pp = map(bad) do model
    p1, ε = hquadrature(-6, 6; abstol, maxevals) do pref
        posterior(model, Observation(true, 3.), pref)
    end

    p2, ε = hquadrature(-6, 6; abstol, maxevals) do pref
        posterior(model, Observation(true, 9.), pref)
    end
    p1, p2
end

# %% --------

findall(R.AA .+ R2.AA .> 101)
sum(R.BC .+ R2.BC .> 101)
sum(R.BA .+ R2.BA .> 101)
findall(R.AC .+ R2.AC .> 101)

maximum(d_ac .- d_ba)
maximum(R.AC .- R.BA)

d_ac = R.AC ./ (R.AC .+ R2.AC)
d_ba = R.BA ./ (R.BA .+ R2.BA)

minimum(d_ac .- d_ba)

good = findall(99.9 .< R.AC .+ R2.AC .< 100.1)

i = argmax(R.AC[good] .- R.BA[good])



# %% --------
d_ac = (R.AC .- R2.AC)
d_ba = (R.BA .- R2.BA)








# %% ==================== Back to the stable code ====================

# How often is each choice in the predicted direction?
@assert all(50 .< R.AA .< 100)
@show mean(50 .< R.BC .< 100) # This one is due to numerical integration error, see below
@assert all(50 .< R.BA .< 100)
@assert all(50 .< R.AC .< 100)

# %% --------
bad = findall(R.BC .< 50)

# TODO
for i in bad
    # due to numerical error, these don't sum to one which is why p1 is less than 0.5
    p1 = predict_bc_choice(DDM(θ, β), Observation(false, 3.), Observation(false, 9.))
    p2 = predict_bc_choice(DDM(θ, β), Observation(false, 3.), Observation(false, 9.); choice=false)
    @assert p1 > p2
end

# %% ==================== Experiment 3 ====================


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


predictions = (;Expt_1, Expt_2, Exp_3)

write("results/sensitivity_analysis.json", JSON.json(predictions))

