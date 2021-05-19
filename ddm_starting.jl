Base.@kwdef struct DDMStarting <: Model
    θ::Float64  # threshold
    β::Float64  # drift-rate multiplier
    sz::Float64  # starting point range
end

function build(model::DDMStarting, pref, starting)
    dd = ConstDrift(model.β * pref, dt)
    bb = ConstAsymBounds(model.θ-starting, -model.θ-starting, dt)
    dd, bb
end

"p(rt, choice | pref, starting)"
function likelihood(model::DDMStarting, obs::Observation, pref, starting)
    dd, bb = build(model, pref, starting)
    rt_pdf = obs.choice ? pdfu : pdfl   # RT pdf for upper (choice=1) or lower threshold
    rt_pdf(dd, bb, obs.rt)
end
"p(rt, choice | pref) marginalzing over starting point"
function likelihood(model::DDMStarting, obs::Observation, pref)
    hquadrature(-model.sz, model.sz) do starting
        likelihood(model, obs, pref, starting)
    end |> first
end

function simulate(model::DDMStarting, pref, starting=rand(Uniform(-model.sz, model.sz)))
    dd, bb = build(model, pref, starting)
    rt, choice = rand(sampler(dd, bb))
    Observation(choice, rt)
end
