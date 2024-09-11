import numpy as np
import math

#f(x)
def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)

#f'(x)
def grad(x):
    return 6*x + 2 + 4*np.cos(x)

def gradient_descent(x0, eta, limit, loop_max):
    x = x0
    for i in range(loop_max):
        x = x - eta*grad(x)
        if abs(grad(x)) < limit:
            print("\nConvergence at loop no:", i)
            return np.round(x, 6), i
    print("\nexecute failed") # can not find convergence point after 10000 loops 
    return np.round(x, 6), loop_max

# Define parameters
x_start = -1       # x0
eta = 0.01          # learning rate
limit = 1e-3       # convergence threshold |f'(x)| < 1e-3
loop_max = 10000   # max loop executable     

#Main
x_min, loop_total = gradient_descent(x_start, eta, limit, loop_max)
print("\nX value at the smallest point:", x_min)
print("\nCost function value with x_min:", np.round(cost(x_min), 6))