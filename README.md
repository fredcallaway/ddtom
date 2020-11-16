# Drift diffusion theory of mind

Models of people's inferences about other's preferences based on reaction times. 

## Setup
This runs on julia 1.4 (and presumably later versions). You need to install the following dependencies:

`] add JSON Dates Glob ProgressMeter`

## Important files
- model.jl: the basic model, supports experiment 1 and 3 predictions
- experiment2.jl: additional code to predict a new choice based on previous choices
- main.jl: you should be able to just run this and reproduce the results, fitted_predictions.json. Note that if you haven't precomputed the experiment 2 predictions, this will take a long time on one core.
