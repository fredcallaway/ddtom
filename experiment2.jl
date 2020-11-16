"
Computes predictions for Experiment 2, in which people observe two choices
involving three items and make a prediction about a choice between the two
items that have not yet been chosen between.

Notation:
    - a, b, and c are the values of three distinct items
    - θ is the DDM threshold (higher is more cautious)
    - σ is the standard deviation of the prior on value
"

include("model.jl")
using Cubature
const BOUNDS = ([-3., -3, -3], [3., 3, 3])  # a, b, c
const MAX_RT = 20.
using DataStructures: OrderedDict

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
    Z, ε = hcubature(score, BOUNDS...)
    
    function posterior(a, b, c)
        score((a, b, c)) / Z
    end
end

"Probability of choosing b over c given observed choices and rts for (a vs. b) and (a vs. c) "
function predict_bc_choice(ab_trial, bc_trial; σ, θ)
    post = make_abc_posterior(ab_trial, bc_trial; σ=σ, θ=θ)
    bounds = deepcopy(BOUNDS); push!(bounds[1], 0); push!(bounds[2], MAX_RT)

    Z, ε = hcubature(bounds..., abstol=1e-5, maxevals=10^6) do (a,b,c,rt)
        post(a,b,c) * likelihood((rt, 1), b - c, θ)
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
        :AA => [(fast, 1), (slow, 1)],  # < 0.5 because b is more below a than c
        :BC => [(fast, 2), (slow, 2)],  #  > 0.5 because b is more above a than c
        :BA => [(fast, 2), (slow, 1)],  # a > b and a << c
        :AC => [(fast, 1), (slow, 2)],  # a >> b and a < c
    )
end

function exp2_predictions(σ, θ)
    map(collect(define_trials())) do (k, trials)
        k => predict_bc_choice(trials...; σ=σ, θ=θ)
    end |> Dict
end
