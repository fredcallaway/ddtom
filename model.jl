# include("model.jl")
using DiffModels
using Statistics
using Optim
using Cubature
using Distributions

# Allow currying for functional fun
Base.reduce(f::Function) = x -> reduce(f, x)
Base.map(f::Function) = x -> map(f, x)

const dt = 1e-3  # temporal granularity of DDM
Trial = Tuple{Real, Int}  # RT, choice (0/1)

"p(rt, choice | drift, threshold)"
function likelihood(trial::Trial, drift, threshold)
    rt, choice = trial
    dd = ConstDrift(drift, dt)
    bb = ConstSymBounds(threshold, dt)
    rt_pdf = choice == 1 ? pdfu : pdfl   # RT pdf for upper (choice=1) or lower threshold
    rt_pdf(dd, bb, rt)
end

function sample_choice_rt(drift, threshold)
    dd = ConstDrift(drift, dt)
    bb = ConstSymBounds(threshold, dt)
    rand(sampler(dd, bb))
end

"p(drift)"
function prior(drift, σ)
    pdf(Normal(0, σ * √2), drift)  # N(0, σ) - N(0, σ)
end

"unnormalized p(drift, threshold | rt, choice)"
function posterior(trial::Trial, drift, threshold, σ)
    likelihood(trial, drift, threshold) * prior(drift, σ)
end

"argmax p(drift | rt, choice, threshold)"
function MAP_drift(trial::Trial, threshold, σ)
    res = optimize(-100, 100) do drift
        -log(posterior(trial, drift, threshold, σ))
        # -(log(prior(drift)) + log(likelihood(trial, drift, threshold)))
    end
    res.minimizer
end

function posterior_mean_drift(trial::Trial, threshold, σ)
    normalizing_constant = hquadrature(-10σ, 10σ) do drift
        posterior(trial, drift, threshold, σ)
    end |> first
    hquadrature(-10σ, 10σ) do drift
        posterior(trial, drift, threshold, σ) / normalizing_constant * drift
    end |> first
end

"argmax p(drift, threshold | rt, choice)"
function MAP_drift_threshold(trial::Trial; max_threshold=10., max_drift=10.)
    lower = [-max_drift, 1e-3]; upper = [max_drift, max_threshold]
    init = @. lower + $rand(2) * (upper - lower)  # randomly initialize within the bounds
    res = optimize(lower, upper, init, Fminbox(LBFGS())) do (drift, threshold)
        -log(posterior(trial, drift, threshold))
    end
    (drift=res.minimizer[1], threshold=res.minimizer[2])
end

# for convenience, assume choice is 1 when you give only an RT
MAP_drift(rt::Real, θ, σ) = MAP_drift((rt, 1), θ, σ)
MAP_drift_threshold(rt::Real, σ) = MAP_drift_threshold((rt, 1), σ)
posterior_mean_drift(rt::Real, θ, σ) = posterior_mean_drift((rt, 1), θ, σ)



