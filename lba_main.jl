using Sobol
using ProgressMeter
using SplitApplyCombine
using Printf

@everywhere begin
    include("fitting.jl")
    include("lba_model.jl")
end

# %% --------
box = Box(
    β = (1, 10, :log),
    β0 = (1, 10, :log),
    θ = (10, 100, :log),
    A = (10, 100, :log),
    sv = 1,
)

@everywhere function full_loss(model)
    reasonable = data_plausible(model) && reasonable_accuracy(model) && low_nochoice_rate(model)
    loss1 = loss2 = missing
    if reasonable 
        loss1 = exp1_loss(model)
        if !isnan(loss1)
            loss2 = exp2_loss(model)
        end
    end
    (;reasonable, loss1, loss2)
end


# %% ==================== Grid ====================

prms = grid(10, box);
models = map(prms) do x
    LBA(x...)
end;
grid_loss = @showprogress pmap(full_loss, models)

using Serialization
serialize("tmp/lba_grid", (;box, models, grid_loss))


# %% --------
using Serialization
using Plots.Measures

include("figure.jl")
box, models, grid_loss = deserialize("tmp/lba_grid");

L = map(grid_loss) do g
    g.loss1 + g.loss2
end;

L[ismissing.(L)] .= Inf
L[isnan.(L)] .= Inf

ndim = 4
juxt(fs...) = x -> Tuple(f(x) for f in fs)
Base.dropdims(idx::Int...) = X -> dropdims(X, dims=idx)
varnames = free(box)

function best(X::Array, dims...; ymax=Inf)
    drop = [i for i in 1:ndim if i ∉ dims]
    B = minimum(X; dims=drop)
    b = permutedims(B, [dims...; drop]) |> dropdims((length(dims)+1:ndim)...)
    b[b .> ymax] .= NaN
    b
end

function get_ticks(i)
    return 1:10, 1:10
    idx = 1:7
    vals = round.(G.dims[i][2]; sigdigits=2)
    idx[1:3:end], vals[1:3:end]
end

function plot_grid(X; ymax=Inf)
    mins, maxs = invert([juxt(minimum, maximum)(best(X, i, j)) for i in 1:ndim for j in i+1:ndim])
    lims = [minimum(mins), maximum(maxs)]

    if ymax != Inf
        lims[2] = ymax
    end

    P = map(1:ndim) do i
        map(1:ndim) do j
            if i == j
                plot(best(X, i), xlabel=varnames[i], ylims=lims, xticks=get_ticks(i))
            elseif i < j
                plot(axis=:off, grid=:off)
            else
                heatmap(best(X, i, j; ymax=ymax),
                    xlabel=varnames[j],
                    ylabel=varnames[i],
                    xticks=get_ticks(j),
                    yticks=get_ticks(i),
                    colorbar=false, clim=Tuple(lims),  aspect_ratio = 1)
            end
        end
    end |> flatten

    plot(P..., size=(1100, 1000), right_margin=4mm)
end

figure() do
    plot_grid(L)
end


# %% ==================== Sobol ====================

N = 100000
xs = Iterators.take(SobolSeq(n_free(box)), N) |> collect
models = map(xs) do x
    LBA(;box(x)...)
end
sobol_loss = @showprogress pmap(full_loss, models)

# %% ====================  ====================
y = map(sobol_loss) do x
    l = √(x.loss1 + x.loss2)
    ismissing(l) ? NaN : l
end

mean(filter(!isnan, y))
y[ismissing.(y)] .= 
mean(skipmissing(y))

# loss12[isnan.(loss12)] .= maximum()

y[isnan.(y)] .= Inf

min_y, min_i = findmin(y)

using Optim


# %% --------

model = models[min_i]
α = optimize_α(model)

θlo, θhi = optimize_θs(model, α)



# %% ==================== Save predictions ====================

predictions = Dict(
    "Expt_1" => Dict(exp1_keys .=> exp1_predict(model, α)),
    "Expt_2" => Dict(exp2_keys .=> exp2_predict(model)),
    "Exp_3" => Dict(exp3_keys .=> exp3_predict(model, θlo, θhi, α))
)

write("results/lba_fitted_predictions.json", JSON.json(predictions))


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


# # %% ==================== GP minimize ====================
# exp2_predict(model) = rescale(exp2_predictions(model))

# x = exp2_predict(model)

# include("gp_min.jl")
# result_gp = gp_minimize(length(box); iterations=1000, verbose=true) do x
#     model = LBA(;box(x)...)
#     e1_loss = optimize(0, 500) do α
#         sse(exp1_predict(model, α), exp1_targets)
#     end |> Optim.minimum

#     e2_loss = sse(exp2_predict(model), exp2_targets)

#     √(e1_loss + e2_loss) / 10
# end


