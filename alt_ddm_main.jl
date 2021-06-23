using ProgressMeter
using SplitApplyCombine
using Distributed
using Serialization

@everywhere begin
    include("fitting.jl")
    include("ddm_starting.jl")
    check_reasonable(model::DDMStarting) = model.sz < model.θ && check_reasonable_base(model)
end

# %% ==================== With starting points ====================

@everywhere box = Box(
    β = (.01, 10, :log),
    θ = (0.1, 100, :log),
    sz = (0.1, 100, :log)
)
models = sample_models(DDMStarting, box)
is_reasonable = @showprogress pmap(check_reasonable, models)
loss = @showprogress pmap(exp1_loss, models[is_reasonable])
model = models[is_reasonable][argmin(loss)]
serialize("fits/ddm_starting", model)

# %% ==================== Exp1 only ====================

# this doesn't really belong here but it's used in the starting point figure so...
@everywhere box = Box(
    β = (.01, 10, :log),
    θ = (0.1, 100, :log),
)
models = sample_models(DDM, box)
is_reasonable = @showprogress pmap(check_reasonable, models)
loss = @showprogress pmap(exp1_loss, models[is_reasonable])
model = models[is_reasonable][argmin(loss)]
serialize("fits/ddm_exp1", model)
