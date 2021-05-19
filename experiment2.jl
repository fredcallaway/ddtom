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
const MAX_SD = 4
const MAX_RT = 240.
using DataStructures: OrderedDict
using Interpolations

function make_bounds(;rt=false)
    lo = -MAX_SD .* ones(3)
    hi = MAX_SD .* ones(3)
    if rt
        push!(lo, 0.); push!(hi, MAX_RT)
    end
    lo, hi
end

function hcubature_m(f::Function, fdim::Integer, xmin, xmax; kws...)
    hcubature(fdim, f, xmin, xmax;  kws...)
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

"Computes the unnormalized posterior distribution for abc_likelihood"
function abc_posterior(model::Model, ab_trial::Observation, bc_trial::Observation, a, b, c)
    # the 1e7 improves numerical stability (I think)
    1e7 * abc_prior(a, b, c) * abc_likelihood(model, ab_trial, bc_trial, a, b, c)
end

"Approximates p(choice | pref) with a dense linear interpolation"
function make_choice_curve(model; choice=true)
    x = .01:.01:2MAX_SD
    p = map(x) do pref
        v, ε = hquadrature(0, MAX_RT) do rt
            likelihood(model, Observation(choice, rt), pref)
        end
        @assert ε < 1e-8
        v
    end
    LinearInterpolation(-2MAX_SD:.01:2MAX_SD, [reverse(1 .- p); 0.5; p])
end

"Probability of choosing b over c given observed choices and rts for (a vs. b) and (a vs. c) "
function predict_bc_choice(model, ab_trial, bc_trial; choice=true, reltol=1e-6, maxevals=10^7)
    # to reduce the dimensionality of the integral, we precompute choice probability
    # for all possible preferences (marginalizing over RT)
    p_choose = make_choice_curve(model; choice)

    Z, ε = hcubature_m(3, make_bounds()...; reltol, maxevals) do (a, b, c), out
        p_vals = abc_posterior(model, ab_trial, bc_trial, a, b, c)
        p_choice = p_choose(b - c)
        out[1] = p_vals * p_choice
        out[2] = p_vals * (1 - p_choice)
        out[3] = p_vals
        # Note: we are computing redundant information here because out[3] = out[1] + out[2]
        # For some reason that I do not even remotely understand, this is necessary for
        # the integral to converge correctly. Apparently, getting below the tolerance on
        # just two of the values is not sufficient.
    end
    pb, pc, normalizer = Z
    @assert (pb + pc) ≈ normalizer # we're not crazy

    ε_choice = ε[1] / normalizer
    if ε_choice > 1e-4
        @warn "predict_bc_choice: integral did not converge" Tuple(Z) Tuple(ε) model ab_trial bc_trial
        return NaN
    end
    
    return pb / normalizer
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
