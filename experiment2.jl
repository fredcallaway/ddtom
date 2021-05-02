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

include("model.jl")
using Cubature
const MAX_SD = 10
const MAX_RT = 240.
using DataStructures: OrderedDict

function make_bounds(σ; rt=false)
    lo = -10σ .* ones(3)
    hi = 10σ .* ones(3)
    if rt
        push!(lo, 0.); push!(hi, MAX_RT)
    end
    lo, hi
end

"Likelihood of observed choice and RT for choices between (a and b) and (a and c)"
function abc_likelihood(ab_trial::Trial, bc_trial::Trial, a, b, c, θ)
    likelihood(ab_trial, a - b, θ) * likelihood(bc_trial, a - c, θ)
end

"Prior probability of values a, b, c"
function abc_prior(a, b, c; σ)
    d = Normal(0, σ)
    pdf(d, a) * pdf(d, b) * pdf(d, c)
end

"Computes the posterior distribution for abc_likelihood"
function make_abc_posterior(ab_trial, bc_trial; σ, θ)
    # unnormalized posterior
    function score((a, b, c))
        abc_prior(a, b, c; σ=σ) * abc_likelihood(ab_trial, bc_trial, a, b, c, θ)
    end
    
    # estimate the normalizing constant
    Z, ε = hcubature(score, make_bounds(σ)...)
    
    function posterior(a, b, c)
        score((a, b, c)) / Z
    end
end

"Probability of choosing b over c given observed choices and rts for (a vs. b) and (a vs. c) "
function predict_bc_choice(ab_trial, bc_trial; σ, θ, choice=1)
    post = make_abc_posterior(ab_trial, bc_trial; σ=σ, θ=θ)

    Z, ε = hcubature(make_bounds(σ; rt=true)..., abstol=1e-5, maxevals=10^7) do (a,b,c,rt)
        post(a,b,c) * likelihood((rt, choice), b - c, θ)
    end
    if ε > 1e-3
        @error "Integral did not converge" ε σ θ
        return NaN
    end
    return Z
end

function define_trials(fast=3., slow=9.)
    # first choice between A and B, second between A and C
    OrderedDict(
        :AA => [(fast, 1), (slow, 1)],  # a >> b  &  a > c  =>  c > b
        :BC => [(fast, 2), (slow, 2)],  # b >> a  &  c > a  =>  b > c
        :BA => [(fast, 2), (slow, 1)],  # b >> a  &  a > c  =>  b >> c
        :AC => [(fast, 1), (slow, 2)],  # a >> b  &  c > a  =>  c >> b
    )
end

function exp2_predictions(σ, θ)
    map(collect(define_trials())) do (k, trials)
        k => predict_bc_choice(trials...; σ=σ, θ=θ)
    end |> Dict
end
