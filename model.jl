# include("model.jl")
using DiffModels
using Statistics
using Optim
using QuadGK
using Distributions

# Allow currying for functional fun
Base.reduce(f::Function) = x -> reduce(f, x)
Base.map(f::Function) = x -> map(f, x)

const dt = 1e-3  # temporal granularity of DDM
Trial = Tuple{Float64, Int}  # RT, choice (0/1)

"p(rt, choice | drift, threshold)"
function likelihood(trial::Trial, drift, threshold)
    rt, choice = trial
    dd = ConstDrift(drift, dt)
    bb = ConstSymBounds(threshold, dt)
    rt_pdf = choice == 1 ? pdfu : pdfl   # RT pdf for upper (choice=1) or lower threshold
    rt_pdf(dd, bb, rt)
end

"p(drift)"
function prior(drift)
    pdf(Normal(0, √2), drift)  # N(0, 1) - N(0, 1)
end

"unnormalized p(drift, threshold | rt, choice)"
function posterior(trial::Trial, drift, threshold)
    likelihood(trial, drift, threshold) * prior(drift)
end

"argmax p(drift | rt, choice, threshold)"
function MAP_drift(trial::Trial, threshold)
    res = optimize(-10, 10) do drift
        -log(posterior(trial, drift, threshold))
        # -(log(prior(drift)) + log(likelihood(trial, drift, threshold)))
    end
    res.minimizer
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
MAP_drift(rt::Float64, threshold) = MAP_drift((rt, 1), threshold)
MAP_drift_threshold(rt::Float64) = MAP_drift_threshold((rt, 1))



