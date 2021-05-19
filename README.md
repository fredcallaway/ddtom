# Code for "Rational preference inference from response time"

Models of people's inferences about other's preferences based on reaction times. 

## Setup
This runs on julia 1.6 (and presumably later versions). You need to install the following dependencies:

`] add JSON ProgressMeter Sobol SplitApplyCombine https://github.com/DrugowitschLab/DiffModels.jl Statistics Optim Cubature Distributions`

## Key files
- model.jl: the basic model, supports experiment 1 and 3 predictions
- experiment2.jl: additional code to predict a new choice based on previous choices
- fitting.jl: base code to fit models to the plotse
- main.jl: fits the DDM model and generates predictions.

The code is not well documented. Please contact me (fredcallaway@princeton.edu) if you have any questions!
