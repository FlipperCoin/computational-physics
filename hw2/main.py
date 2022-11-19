import numpy as np
import matplotlib.pyplot as plt

from numalg import euler_method

def harmonic_oscillator_euler(T, N, v0, x0):
    return euler_method(T, N, lambda x: np.array([x[1], -x[0]]), np.array([x0, v0]))

def harmonic_oscillator_rk2(T, N, v0, x0):
    pass

def harmonic_oscillator_rk4(T, N, v0, x0):
    pass

T = 2 * np.pi
N = 20

x = harmonic_oscillator_euler(T, N, 0, 1)

plt.plot(np.linspace(0, T, N+1), x[:, 0])
plt.xlabel(r't $\left[ sec \right]$')
plt.ylabel(r'x $\left[ m \right]$')
plt.grid()
plt.show()
