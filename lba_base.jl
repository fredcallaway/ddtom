# adapted from https://github.com/smfleming/LBA/

using StatsFuns
using Distributions
using Cubature

# Run a single trial of the LBA model (Brown & Heathcote, 2008, Cog
# Psychol)
#
# Inputs:
#
# A = range of uniform distribution U[0,A] from which starting point k is
# drawn
# b = bound
# v = vector of drift rates
# sv = standard deviation of drift rate
# t0 = non-decision time
# N = number of response options
#
# Outputs:
#
# choice = scalar from 1:N indicating response chosen by model
# RT = reaction time in ms
function LBA_trial(A, b, v, t0, sv, N)
    allRT = map(v) do vi
        # Get drift rate
        d = rand(Normal(vi, sv))
        if d < 0
            return Inf  # can't hit threshold (assuming A < b)
        end
        # Get starting point
        k = rand() * A
        # Get time to threshold
        t = (b-k)/d
        # Add on non-decision time
        t0 + t
    end
    
    # Get choice & confidence
    rt, choice = findmin(allRT)
    if rt == Inf
        choice = 0
    end
    (;rt, choice)
end

# Get PDF of first passage time of ith accumulator in LBA model
function LBA_tpdf(t, A, b, v, sv)
    g = (b-A-t*v)/(t*sv)
    h = (b-t*v)/(t*sv)
    (-v*normcdf(g) + sv*normpdf(g) + v*normcdf(h) - sv*normpdf(h))/A
end

# Get CDF of first passage time of ith accumulator in LBA model
function LBA_tcdf(t, A, b, v, sv)
    g = (b-A-t*v)/(t*sv); # chizumax
    h = (b-t*v)/(t*sv);  # chizu
    i = b-t*v; # chiminuszu
    j = i-A; # xx

    tmp1 = t*sv*(normpdf(g)-normpdf(h));
    tmp2 = j*normcdf(g) - i*normcdf(h);

    F = 1 + (tmp1 + tmp2)/A;
end

# Generates choice probability for responses on node #1 by numerical
# integration of LBA_n1PDF
function LBA_n1CDF(A, b, v, sv)
    hquadrature(0, 100) do rt
        # note: the matlab has A+b where I have A but I think that's incorrect.
        LBA_n1PDF(rt, A, b, v, sv)
    end |> first
    # cdf = quad(@(t)LBA_n1PDF(t,A,A+b,v,sv),1,100000);
end

# https://github.com/smfleming/LBA/blob/master/LBA_n1PDF.m
# Generates defective PDF for responses on node #1 (ie. normalised by
# probability of this node winning race)
function LBA_n1PDF(t, A, b, v, sv)
    N = size(v,2);

    # if N > 2
    #     for i = 2:N
    #         tmp(:,i-1) = LBA_tcdf(t,A,b,v(:,i),sv);
    #     end
    #     G = prod(1-tmp,2);
    # else
    G = 1-LBA_tcdf(t,A,b,v[2],sv)
    # end
    pdf = G.*LBA_tpdf(t,A,b,v[1],sv)
end

