## Dependencies
numpy  
matplotlib

## How to run
Make sure . /Price is in the same directory as linear_regression.py, just open your IDE and run `linear_regression.py` it directly. 

## Output
### Visualization image
The orange points are the original samples, and the blue lines are the solutions under MSE minimization fitted with a straight line.  
### Print text
Gradient descent method iteration steps and errors are output as `step:xxx, loss:xxx` once every 10 steps. The final linear equations predicted by the analytical method and GD method and the prediction of the house price in Nanjing in 2014 are output respectively.  

## Hyperparameters
lr: 1e-9  
Initial θ: [-1600, 2]
