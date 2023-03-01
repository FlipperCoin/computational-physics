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

def oscillator_E(x):
    return (1/2)*x[:, 0]**2 + (1/2)*x[:, 1]**2

def oscillator_deltaE(E):
    return np.abs(E_analytic - E) / E_analytic

E_analytic = oscillator_E(x_analytic)

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
    e_axs.plot(tn, oscillator_deltaE(oscillator_E(x)), '.', label=method_name)

    # Nmin = 1
    # while np.abs((1/2)-E(method_func(T, Nmin, 0, 1))[-1])/(1/2) >= 0.01:
    #     Nmin += 1
    # print(rf'Nmin for {method_name} is: {Nmin}')

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

def two_body_f(x):
    return np.array([x[2], x[3], -x[0] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2)), -x[1] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2))])

def two_body_euler(T, N, x0, y0, vx0, vy0):
    return euler_method(T, N, two_body_f, np.array([x0, y0, vx0, vy0]))

def two_body_rk4(T, N, x0, y0, vx0, vy0):
    return rk4_method(T, N, two_body_f, np.array([x0, y0, vx0, vy0]))

T = 2*np.pi
N = 15
tn = np.linspace(0, T, N+1)

methods = [('RK4', two_body_rk4),
           ('euler', two_body_euler)]

x_fig = plt.figure(3)
x_axs = x_fig.subplots()
y_fig = plt.figure(4)
y_axs = y_fig.subplots()
polar_fig = plt.figure(5)
polar_axs = polar_fig.subplots(subplot_kw={'projection': 'polar'})
# e_fig = plt.figure(2)
# e_axs = e_fig.subplots()

tn2 = np.linspace(0, T, 100)
x_analytic = np.column_stack((np.cos(tn2), np.sin(tn2), -np.sin(tn2), np.cos(tn2)))

def orbit_E(x):
    return (1 / 2) * (x[:, 2] ** 2 + x[:, 3] ** 2) - (1/np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))

E_analytic = orbit_E(x_analytic)

for method_name, method_func in methods:
    x = method_func(T, N, 1, 0, 0, 1)

    x_axs.plot(tn, x[:, 0], '.', label=method_name)
    y_axs.plot(tn, x[:, 1], '.', label=method_name)
    polar_axs.plot(np.arctan2(x[:, 1], x[:, 0]), np.sqrt(x[:, 0]**2+x[:, 1]**2), '.', label=method_name)

    relE = np.abs(E_analytic[-1] - orbit_E(x)[-1]) / np.abs(E_analytic[-1])
    print(f'relE in {method_name} is {relE}')

x_axs.plot(tn2, x_analytic[:, 0], label='analytic')
y_axs.plot(tn2, x_analytic[:, 1], label='analytic')
polar_axs.plot(np.arctan2(x_analytic[:, 1], x_analytic[:, 0]), np.sqrt(x_analytic[:, 0]**2+x_analytic[:, 1]**2), label='analytic')

x_axs.set_xlabel(r't')
x_axs.set_ylabel(r'x')
x_axs.legend()
x_axs.grid()
x_fig.savefig('numeric_two_body_circ_xt.png')
x_fig.show()

y_axs.set_xlabel(r't')
y_axs.set_ylabel(r'y')
y_axs.legend()
y_axs.grid()
y_fig.savefig('numeric_two_body_circ_yt.png')
y_fig.show()

polar_axs.set
polar_axs.legend()
polar_axs.grid(True)
polar_axs.set_rticks([1, 2, 3, 4])
polar_fig.savefig('numeric_two_body_circ_polar.png')
polar_fig.show()
