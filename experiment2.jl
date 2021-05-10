"
Computes predictions for Experiment 2, in which people observe two choices
involving three items and make a prediction about a choice between the two
items that have not yet been chosen between.

Notation:
    - a, b, and c are the values of three distinct items (after scaling by β)
    - θ is the DDM threshold (higher is more cautious)
    - σ is the drift rate multiplier (higher is stronger evidence)
      (it can also be interpreted as the standard deviation of the prior on value)
"

using Cubature
const MAX_SD = 10
const MAX_RT = 240.
using DataStructures: OrderedDict

function make_bounds(;rt=false)
    lo = -10 .* ones(3)
    hi = 10 .* ones(3)
    if rt
        push!(lo, 0.); push!(hi, MAX_RT)
    end
    lo, hi
end

"Likelihood of observed choice and RT for choices between (a and b) and (a and c)"
function abc_likelihood(model::Model, ab_trial::Observation, bc_trial::Observation, a, b, c)
    likelihood(model, ab_trial, a - b) * likelihood(model, bc_trial, a - c)
end

"Prior probability of values a, b, c"
function abc_prior(a, b, c)
    d = Normal(0, 1)
    pdf(d, a) * pdf(d, b) * pdf(d, c)
end

"Computes the posterior distribution for abc_likelihood"
function make_abc_posterior(model, ab_trial, bc_trial)
    # unnormalized posterior
    function score((a, b, c))
        abc_prior(a, b, c) * abc_likelihood(model, ab_trial, bc_trial, a, b, c)
    end
    
    # estimate the normalizing constant
    Z, ε = hcubature(score, make_bounds()..., abstol=1e-5, maxevals=10^7)
    if ε > 1e-3
        @error "abc_posterior: integral did not converge" model ab_trial bc_trial
        return NaN
    end
    
    function posterior(a, b, c)
        score((a, b, c)) / Z
    end
end

"Probability of choosing b over c given observed choices and rts for (a vs. b) and (a vs. c) "
function predict_bc_choice(model, ab_trial, bc_trial; choice=true)
    post = make_abc_posterior(model, ab_trial, bc_trial)

    Z, ε = hcubature(make_bounds(rt=true)..., abstol=1e-5, maxevals=10^7) do (a,b,c,rt)
        post(a,b,c) * likelihood(model, Observation(choice, rt), b - c)
    end
    if ε > 1e-3
        @warn "predict_bc_choice: integral did not converge" model ab_trial bc_trial
        return NaN
    end
    return Z
end

function define_trials(fast=3., slow=9.)
    # first choice between A and B, second between A and C
    OrderedDict(
        :AA => [Observation(true, fast), Observation(true, slow)],  # a >> b  &  a > c  =>  c > b
        :BC => [Observation(false, fast), Observation(false, slow)],  # b >> a  &  c > a  =>  b > c
        :BA => [Observation(false, fast), Observation(true, slow)],  # b >> a  &  a > c  =>  b >> c
        :AC => [Observation(true, fast), Observation(false, slow)],  # a >> b  &  c > a  =>  c >> b
    )
end

function exp2_predictions(model; choice=true)
    map(collect(define_trials())) do (k, trials)
        k => predict_bc_choice(model, trials...; choice)
    end |> Dict
end
