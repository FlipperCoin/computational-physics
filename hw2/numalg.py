import numpy as np

def euler_step(x, f, tau):
    return x + tau * f(x)

def euler_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps+1, d))
    x[0] = x0

    for n in range(0, steps):
        x[n+1] = euler_step(x[n], f, tau)

    return x

def rk2_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps + 1, d))
    x[0] = x0

    for n in range(0, steps):
        x[n + 1] = x[n] + tau * f(euler_step(x[n], f, tau/2))

    return x

def rk4_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps + 1, d))
    x[0] = x0

    for n in range(0, steps):
        F1 = f(x[n])
        F2 = f(x[n] + (1/2)*tau*F1)
        F3 = f(x[n] + (1/2)*tau*F2)
        F4 = f(x[n] + tau*F3)
        x[n + 1] = x[n] + (1/6)*tau*(F1+2*F2+2*F3+F4)

    return x

