include("figure.jl")
include("fitting.jl")
include("lba_model.jl")
using Serialization

# %% --------

ddm = deserialize("fits/ddm")
lba = deserialize("fits/lba")
ddm_exp1 = deserialize("fits/ddm_exp1")
ddm_starting = deserialize("fits/ddm_starting")
ddm_starting_fixed = DDMStarting(;ddm_exp1.β, ddm_exp1.θ, sz=ddm_exp1.θ)

# %% --------

function pretty_print(m::Model, tag="")
    println(typeof(m), tag)
    for k in fieldnames(typeof(m))
        println("  ", rpad(k, 4), round(getfield(m, k); sigdigits=4))
    end
    println()
end

function compact_print(m::Model, tag="")
    parts = map(fieldnames(typeof(m))) do k
        "$k = " * string(round(getfield(m, k); sigdigits=4))
    end
    println("(", join(parts, ", "), ")")
end

pretty_print(ddm)
pretty_print(ddm_exp1, "-exp1")
pretty_print(ddm_starting)
pretty_print(lba)

compact_print(ddm)
compact_print(ddm_exp1, "-exp1")
compact_print(ddm_starting)
compact_print(lba)

# %% --------

function plot_curve!(model, color, label=string(typeof(model)))
    x = 0.1:0.01:10
   plot!(x, optimize_α(model) .* posterior_mean_pref.([model], x); color, label)
end

function plot_data!()
    scatter!([3, 5, 7, 9], exp1_targets, color=:black,
        xaxis=("Response time (secs)", (0,10), 1:2:9), 
        yaxis=("Inferred Preference", (0,Inf)),
        label="Data"
    )
end

figure("starting_points"; pdf=true) do
    plot_data!()
    plot_curve!(ddm_exp1, 1, "Vanilla DDM")
    plot_curve!(ddm_starting_fixed, 2, "DDM with fixed large starting point range")
    plot_curve!(ddm_starting, 3, "DDM with fitted starting point range")
end

figure("ddm_lba_curve"; pdf=true) do
    plot_data!()
    plot_curve!(ddm, 1)
    plot_curve!(lba, 2)
end


