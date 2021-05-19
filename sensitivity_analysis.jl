using Sobol
using ProgressMeter
using SplitApplyCombine
@everywhere using Random
using Distributed

@everywhere include("fitting.jl")
include("figure.jl")

DISABLE_PLOTTING = false
pyplot()
mkpath("figs")

# %% ==================== Identify reasonable paramaters ====================
@everywhere Random.seed!(123)

big_βs = 10 .^ (-2:.1:1)
big_θs = 10 .^ (-1:0.1:2)
big_grid = collect(Iterators.product(big_βs, big_θs))

# plaus = @showprogress map(big_grid) do (β, θ)
#     data_plausible(DDM(;β, θ))
# end

rt = @showprogress pmap(big_grid) do (β, θ)
    reasonable_rt(DDM(;β, θ))
end

acc = @showprogress pmap(big_grid) do (β, θ)
    reasonable_accuracy(DDM(;β, θ))
end

reasonable = big_grid[acc .& rt]
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
    make_heat(rt, "#81C0FF")
    make_heat(acc, "#FFE783")
    make_heat(acc .& rt, "#7DE87D")
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

# All choices in the predicted direction
@assert all(50 .< R.AA .< 100)
@assert all(50 .< R.BC .< 100)
@assert all(50 .< R.BA .< 100)
@assert all(50 .< R.AC .< 100)

# Equivalent conditions are equivalent
@assert R.AA ≈ R.BC
@assert R.BA ≈ R.AC

# %% ==================== Experiment 3 ====================

grid3 = collect(Iterators.product(big_βs, big_θs, big_θs))[:]
filter!(grid3) do (β, θlo, θhi)
    θhi > θlo && (β, θlo) in reasonable && (β, θhi) in reasonable
end

# Exp (instead of Expt) was an unfortunate early established standard
Exp_3 = map(grid3) do (β, θlo, θhi)
    model = DDM(;β, θ=NaN)
    α = optimize(0, 500) do α
        exp3_loss(model, θlo, θhi, α)
    end |> Optim.minimizer
    prediction = Dict(exp3_keys .=> exp3_predict(model, θlo, θhi, α))
    (;β, θlo, θhi, α, prediction)
end

# %% --------
# All the differences are in the predicted direction
@assert all(Exp_3) do res
    x = res.prediction
    x["thLo9"] < x["thHi9"] &&
    x["thLo3"] < x["thHi3"] &&
    x["thLo9"] < x["thLo3"] &&
    x["thHi9"] < x["thHi3"]
end

n_interact = map(Exp_3) do res
    x = res.prediction
    x["thHi3"] - x["thHi9"] > x["thLo3"] - x["thLo9"]
end |> sum

println("$n_interact / $(length(Exp_3)) have the interaction" )


# %% --------

predictions = (;Expt_1, Expt_2, Exp_3)

write("results/sensitivity_analysis.json", JSON.json(predictions))

