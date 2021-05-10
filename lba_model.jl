include("lba_base.jl")

Base.@kwdef struct LBA <: Model
    β::Float64  # drift slope
    β0::Float64  # drift intercept (to prevent negative drifts and undefined RTs)
    θ::Float64  # threshold (usually notated b)
    A::Float64  # maximum starting point 
    sv::Float64  # standard deviation of drift noise
end

"p(rt, choice | a, b, threshold)"
function likelihood(model::LBA, obs::Observation, pref)
    drifts = model.β0 .+ model.β .* [pref; -pref]
    if !obs.choice
        reverse!(drifts)  # LBA_n1PDF assumes first item is chosen
    end
    LBA_n1PDF(obs.rt, model.A, model.θ, drifts, model.sv)
end

function simulate(model::LBA, pref)
    drifts = model.β0 .+ model.β .* [pref; -pref]
    LBA_trial(model.A, model.θ, drifts, 0, model.sv, 2)
end
