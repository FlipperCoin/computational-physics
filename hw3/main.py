import numpy as np
import matplotlib.pyplot as plt
from time import time

from numalg import rk4_method, rk4_adaptive_method

def two_body_f(x, t=None):
    return np.array([x[2], x[3], -x[0] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2)), -x[1] / ((x[0] ** 2 + x[1] ** 2) ** (3 / 2))])

def two_body_rk4(T, N, x0, y0, vx0, vy0):
    return rk4_method(T, N, two_body_f, np.array([x0, y0, vx0, vy0]))

T = (17/48)*2*np.pi
N = 50000
tn = np.linspace(0, T, N+1)

x0 = 1
y0 = 0
vx0 = 0
vy0 = 1

f = 0.1
vy0 *= f

E_analytic = np.zeros(N + 1) + (1 / 2) * f ** 2 - 1

methods = [('RK4', two_body_rk4)]

def orbit_E(x):
    return (1 / 2) * (x[:, 2] ** 2 + x[:, 3] ** 2) - (1 / np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2))

def delta(big_step, small_step, _):
    big_E = (1 / 2) * (big_step[2] ** 2 + big_step[3] ** 2) - (1 / np.sqrt(big_step[0] ** 2 + big_step[1] ** 2))
    small_E = (1 / 2) * (small_step[2] ** 2 + small_step[3] ** 2) - (1 / np.sqrt(small_step[0] ** 2 + small_step[1] ** 2))
    return np.abs(big_E - small_E) / np.abs(small_E)

def two_body_rk4_adaptive(T, x0, y0, vx0, vy0):
    return rk4_adaptive_method(T, two_body_f, np.array([x0, y0, vx0, vy0]), delta, eps=0.007)

def plot_elliptic():
    x_fig = plt.figure(3)
    x_axs = x_fig.subplots()
    y_fig = plt.figure(4)
    y_axs = y_fig.subplots()
    polar_fig = plt.figure(5)
    polar_axs = polar_fig.subplots(subplot_kw={'projection': 'polar'})
    # e_fig = plt.figure(2)
    # e_axs = e_fig.subplots()

    tn2 = np.linspace(0, T, 100)

    for method_name, method_func in methods:
        st = time()
        x = method_func(T, N, x0, y0, vx0, vy0)
        ed = time()
        total_time = ed-st

        x_axs.plot(tn, x[:, 0], '.', label=method_name)
        y_axs.plot(tn, x[:, 1], '.', label=method_name)
        polar_axs.plot(np.arctan2(x[:, 1], x[:, 0]), np.sqrt(x[:, 0]**2+x[:, 1]**2), '.', label=method_name)

        relE = np.abs(E_analytic[-1] - orbit_E(x)[-1]) / np.abs(E_analytic[-1])
        print(f'relE in {method_name} is {relE}')
        print(f'total runtime {total_time}')

    x_axs.set_xlabel(r't')
    x_axs.set_ylabel(r'x')
    x_axs.legend()
    x_axs.grid()
    x_fig.savefig('numeric_two_body_elliptic_xt.png')
    x_fig.show()

    y_axs.set_xlabel(r't')
    y_axs.set_ylabel(r'y')
    y_axs.legend()
    y_axs.grid()
    y_fig.savefig('numeric_two_body_elliptic_yt.png')
    y_fig.show()

    polar_axs.legend()
    polar_axs.grid(True)
    polar_axs.set_rticks([0.5, 1, 1.5])
    polar_fig.savefig('numeric_two_body_elliptic_polar.png')
    polar_fig.show()

def max_tau():
    target_epsilon = 0.01

    for method_name, method_func in methods:
        N_chosen = 27000
        granularity = 100
        while True:
            st = time()
            x = method_func(T, N_chosen, x0, y0, vx0, vy0)
            ed = time()
            runtime = ed-st

            relE = np.abs(E_analytic[-1] - orbit_E(x)[-1]) / np.abs(E_analytic[-1])
            if relE > target_epsilon:
                N_chosen += granularity
                break

            N_chosen -= granularity

        print(f'for epsilon {target_epsilon}, tau <= {T/N_chosen}, N >= {N_chosen}, runtime {runtime}')

def adaptive_elliptic():
    st = time()
    x, tn = two_body_rk4_adaptive(T, x0, y0, vx0, vy0)
    ed = time()
    total_time = ed - st

    x_fig = plt.figure(3)
    x_axs = x_fig.subplots()
    y_fig = plt.figure(4)
    y_axs = y_fig.subplots()
    polar_fig = plt.figure(5)
    polar_axs = polar_fig.subplots(subplot_kw={'projection': 'polar'})
    e_fig = plt.figure(6)
    e_axs = e_fig.subplots()

    x_axs.plot(tn, x[:, 0], '.', label='adaptive rk4')
    y_axs.plot(tn, x[:, 1], '.', label='adaptive rk4')
    polar_axs.plot(np.arctan2(x[:, 1], x[:, 0]), np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2), '.', label='adaptive rk4')
    e_axs.plot(tn, np.abs(E_analytic[0] - orbit_E(x)) / np.abs(E_analytic[0]), '.', label='adaptive rk4')

    relE = np.abs(E_analytic[-1] - orbit_E(x)[-1]) / np.abs(E_analytic[-1])
    print(f'rk4 adaptive total steps {len(tn)-1}')
    print(f'relE in rk4 adaptive is {relE}')
    print(f'total runtime {total_time}')

    x_axs.set_xlabel(r't')
    x_axs.set_ylabel(r'x')
    x_axs.legend()
    x_axs.grid()
    x_fig.savefig('two_body_elliptic_adaptive_xt.png')
    x_fig.show()

    y_axs.set_xlabel(r't')
    y_axs.set_ylabel(r'y')
    y_axs.legend()
    y_axs.grid()
    y_fig.savefig('two_body_elliptic_adaptive_yt.png')
    y_fig.show()

    polar_axs.legend()
    polar_axs.grid(True)
    polar_axs.set_rticks([0.5, 1, 1.5])
    polar_fig.savefig('two_body_elliptic_adaptive_polar.png')
    polar_fig.show()

    e_axs.set_xlabel(r't')
    e_axs.set_ylabel(r'$\epsilon = \frac{\Delta E}{E}$')
    e_axs.legend()
    e_axs.grid()
    e_fig.savefig('two_body_elliptic_adaptive_error.png')
    e_fig.show()

adaptive_elliptic()
