import numpy as np
import math

#f(x)
def cost(x):
    return 3*x**2 + 2*x + 4*np.sin(x)

#f'(x)
def grad(x):
    return 6*x + 2 + 4*np.cos(x)

def gradient_descent(x0, eta, limit, loop_max):
    x = [x0]
    for i in range(loop_max):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < limit:
            print("\nConvergence at loop no:", i)
            return np.round(x_new, 6), i
        x.append(x_new)
    print("\nexecute failed") # can not find convergence point after 10000 loops 
    return np.round(x_new, 6), loop_max

# Define parameters
x_start = 5       # x0
eta = 0.1          # learning rate
limit = 1e-3       # convergence threshold |f'(x)| < 1e-3
loop_max = 10000   # max loop executable     

#Main
x_min, loop_total = gradient_descent(x_start, eta, limit, loop_max)
print("\nX value at the smallest point:", x_min)
print("\nCost function value with x_min:", np.round(cost(x_min), 6))