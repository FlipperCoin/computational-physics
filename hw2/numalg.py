import numpy as np

def euler_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps+1, d))
    x[0] = x0

    for n in range(0, steps):
        x[n+1] = x[n] + tau*f(x[n])

    return x