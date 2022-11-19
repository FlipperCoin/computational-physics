import numpy as np
import matplotlib.pyplot as plt

from numalg import euler_method, rk2_method, rk4_method

def harmonic_oscillator_euler(T, N, v0, x0):
    return euler_method(T, N, lambda x: np.array([x[1], -x[0]]), np.array([x0, v0]))

def harmonic_oscillator_rk2(T, N, v0, x0):
    return rk2_method(T, N, lambda x: np.array([x[1], -x[0]]), np.array([x0, v0]))

def harmonic_oscillator_rk4(T, N, v0, x0):
    return rk4_method(T, N, lambda x: np.array([x[1], -x[0]]), np.array([x0, v0]))

T = 2 * np.pi
N = 15

methods = [('euler', harmonic_oscillator_euler),
           ('RK2', harmonic_oscillator_rk2),
           ('RK4', harmonic_oscillator_rk4)]

for method_name, method_func in methods:
    x = method_func(T, N, 0, 1)
    plt.plot(np.linspace(0, T, N + 1), x[:, 0], '.', label=method_name)

plt.xlabel(r't $\left[ sec \right]$')
plt.ylabel(r'x $\left[ m \right]$')
plt.legend()
plt.grid()
plt.savefig('numeric_harmonic_oscillator.png')
plt.show()


