"
Basic model for inferring preferences from reaction times by inverting a DDM

Note: σ is called β in the paper. Here we interpret it as a prior over values
rather than a multiplier on the drift rate. These are mathematically equivalent
because x ~ Normal(0, σ) is equivalent to x ~ σ * Normal(0, 1).
"

using DiffModels
using Statistics
using Optim
using Cubature
using Distributions

# Allow currying for functional fun
Base.reduce(f::Function) = x -> reduce(f, x)
Base.map(f::Function) = x -> map(f, x)

const dt = 1e-3  # temporal granularity of DDM

struct Observation
    choice::Bool # choose first
    rt::Float64     # reaction time
end

abstract type Model end
Base.length(::Model) = 1

Base.@kwdef struct DDM <: Model
    θ::Float64  # threshold
    β::Float64  # drift-rate multiplier
end

function build(model::DDM, pref)
    dd = ConstDrift(model.β * pref, dt)
    bb = ConstSymBounds(model.θ, dt)
    dd, bb
end

"p(rt, choice | pref)"
function likelihood(model::DDM, obs::Observation, pref)
    dd = ConstDrift(model.β * pref, dt)
    bb = ConstSymBounds(model.θ, dt)
    rt_pdf = obs.choice ? pdfu : pdfl   # RT pdf for upper (choice=1) or lower threshold
    rt_pdf(dd, bb, obs.rt)
end

function simulate(model::DDM, pref)
    dd, bb = build(model, pref)
    rt, choice = rand(sampler(dd, bb))
    Observation(choice, rt)
end

"p(pref)"
function prior(pref)
    pdf(Normal(0, √2), pref)  # N(0, 1) - N(0, 1)
end

"unnormalized p(pref | rt, choice)"
function posterior(model::Model, obs::Observation, pref)
    likelihood(model, obs, pref) * prior(pref)
end

function posterior_mean_pref(model::Model, obs::Observation)
    normalizing_constant, ε = hquadrature(-10, 10; abstol=1e-5, maxevals=10^7) do pref
        posterior(model, obs, pref)
    end
    if ε > 1e-3
        @error "normalizing_constant: integral did not converge" model obs
        return NaN
    end
    v, ε = hquadrature(-10, 10; abstol=1e-5, maxevals=10^7) do pref
        posterior(model, obs, pref) / normalizing_constant * pref
    end
    if ε > 1e-3
        @error "posterior_mean_pref: integral did not converge" model obs
        return NaN
    end
    v
end

posterior_mean_pref(model::Model, rt::Real) = posterior_mean_pref(model, Observation(true, rt))
