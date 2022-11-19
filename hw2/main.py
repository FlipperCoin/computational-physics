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
N = 10

tn = np.linspace(0, T, N + 1)
x_analytic = np.column_stack((np.cos(tn), -1*np.sin(tn)))

def E(x):
    return (1/2)*x[:, 0]**2 + (1/2)*x[:, 1]**2

def deltaE(E):
    return np.abs(E_analytic - E) / E_analytic

E_analytic = E(x_analytic)

methods = [('RK4', harmonic_oscillator_rk4),
           ('RK2', harmonic_oscillator_rk2),
           ('euler', harmonic_oscillator_euler)]

pos_fig = plt.figure(1)
pos_axs = pos_fig.subplots()
e_fig = plt.figure(2)
e_axs = e_fig.subplots()

for method_name, method_func in methods:
    x = method_func(T, N, 0, 1)

    pos_axs.plot(tn, x[:, 0], '.', label=method_name)
    e_axs.plot(tn, deltaE(E(x)), '.', label=method_name)

    Nmin = 1
    while np.abs((1/2)-E(method_func(T, Nmin, 0, 1))[-1])/(1/2) >= 0.01:
        Nmin += 1
    print(rf'Nmin for {method_name} is: {Nmin}')

tn2 = np.linspace(0, T, 100)
pos_axs.plot(tn2, np.cos(tn2), label='analytic')
pos_axs.set_xlabel(r't $\left[ sec \right]$')
pos_axs.set_ylabel(r'x $\left[ m \right]$')
pos_axs.legend()
pos_axs.grid()
pos_fig.savefig('numeric_harmonic_oscillator_position.png')
pos_fig.show()

e_axs.set_xlabel(r't $\left[ sec \right]$')
e_axs.set_ylabel(r'$\frac{\Delta E}{E}$')
e_axs.legend()
e_axs.grid()
e_fig.savefig('numeric_harmonic_oscillator_energy.png')
e_fig.show()

