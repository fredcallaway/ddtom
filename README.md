# Code for "Rational preference inference from response time"

Models of people's inferences about other's preferences based on response times. 

## Setup
This runs on julia 1.6 (and presumably later versions). You need to install the following dependencies:

`] add JSON ProgressMeter Sobol SplitApplyCombine https://github.com/DrugowitschLab/DiffModels.jl Statistics Optim Cubature Distributions DataStructures Interpolation Plots`

## Key files
- model.jl: the basic model, supports experiment 1 and 3 predictions
- experiment2.jl: additional code to predict a new choice based on previous choices
- fitting.jl: base code to fit models to the plotse
- main.jl: fits the DDM model and generates predictions.
- lba_main.jl: fits the LBA model
- alt_ddm_main.jl: fits DDM to exp1 only and with starting points
- sensitivity_analysis.jl: self-explanatory

The code is not well documented. Please contact me (fredcallaway@gmail.com) if you have any questions!
