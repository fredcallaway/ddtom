"
Computes predictions for Experiment 2, in which people observe two choices
involving three items and make a prediction about a choice between the two
items that have not yet been chosen between.

Notation:
    - x, y, and z are the values of three distinct items
    - θ is the DDM threshold (higher is more cautious)
"

include("model.jl")
using Cubature

const BOUNDS = ([-3, -3, -3, 0.], [3, 3, 3, 3.])  # x, y, z, θ
const MAX_RT = 10.

"Likelihood of observed choice and RT for choices between (x and y) and (x and z)"
function xyxz_likelihood(xy_trial, xz_trial, x, y, z, θ)
    likelihood(xy_trial, x - y, θ) * likelihood(xz_trial, x - z, θ)
end

"Prior probability of values x, y, z and threshold θ"
function xyzθ_prior(x, y, z, θ)
    d_value = Normal(0, 1)
    d_thresh = Exponential(1)
    pdf(d_value, x) * pdf(d_value, y) * pdf(d_value, z) * pdf(d_thresh, θ)
end

"Computes the posterior distribution for xyxz_likelihood"
function make_xyxz_posterior(xy_trial, xz_trial)
    # unnormalized posterior
    function score((x, y, z, θ))
        xyzθ_prior(x, y, z, θ) * xyxz_likelihood(xy_trial, xz_trial, x, y, z, θ)
    end
    
    # estimate the normalizing constant
    Z, ε = hcubature(score, BOUNDS..., abstol=1e-5)
    
    function posterior(x, y, z, θ)
        score((x, y, z, θ)) / Z
    end
end

"Probability of choosing y over z given observed choices and rts for (x vs. y) and (x vs. z) "
function predict_yz_choice(xy_trial, xz_trial)
    post = make_xyxz_posterior(xy_trial, xz_trial)
    bounds = deepcopy(BOUNDS); push!(bounds[1], 0); push!(bounds[2], MAX_RT)
    hcubature(bounds..., abstol=1e-5) do (x,y,z,θ,rt)
        post(x,y,z,θ) * likelihood((rt, 1), y - z, θ)
    end |> first
end


function define_trials(fast=1., slow=2.)
    Dict(
        :XX => [(fast, 1), (slow, 1)],  # < 0.5 because y is more below x than z
        :xx => [(fast, 0), (slow, 0)],  #  > 0.5 because y is more above x than z
        :Xx_fast => [(slow, 1), (fast, 0)],  # x > y and x << z
        :Xx_slow => [(fast, 1), (slow, 0)],  # x >> y and x < z
    )
end

valmap(f, d::Dict) = Dict(k => f(v) for (k, v) in d)

all_trials = define_trials()
predictions = valmap(all_trials) do trials
    predict_yz_choice(trials...)
end

#=
  :xx      => 0.642579
  :Xx_fast => 0.119947
  :XX      => 0.338768
  :Xx_slow => 0.119947

