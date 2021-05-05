using Revise
using CSV
using DataFrames
using Statistics
includet("lba.jl")
include("figure.jl")

rsim = CSV.read("tmp/lba_sim.csv", DataFrame)
# %% --------

A = 0.5; b = 2; v = [1.8, 1.6]; t0 = 0; sv = .1; N = 2

sim = map(1:10000) do i
    LBA_trial(A, b, v, t0, sv, N)
end |> DataFrame

@show mean(sim.choice .== 1);
@show mean(rsim.response .== 1);
@show LBA_n1CDF(A, b, v, sv);

# %% --------
figure() do
    x = 0:.001:2
    y = map(x) do rt
        LBA_n1PDF(rt, A, b, v, sv)
    end
    histogram(sim[sim.choice .== 1, :].rt; normalize=:pdf)
    plot!(x, y, lw=3, xlim=(0, 2))
end
