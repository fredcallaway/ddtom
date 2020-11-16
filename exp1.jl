include("model.jl")
include("figure.jl")

# %% --------
figure("exp1") do 
    rts = (3:.1:9)
    β, θ, σ = 20, 12, .4
    plot(rts, β.*MAP_drift.(rts, θ, σ),
        size=(400,300),
        xaxis="Decision Time", 
        yaxis="Model Inferred Preference", 
        ylim=(0,50),
    )
end
