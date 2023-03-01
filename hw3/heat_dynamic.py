import numpy as np
import matplotlib.pyplot as plt
from time import time

from numalg import ftcs

def solve(omega, t, label):
    T0 = 1
    T_osc = 2*np.pi/omega

    total_time = t
    total_space = 1
    tres = 0.00001
    sres = 0.01

    x = np.arange(total_space+sres, step=sres)
    t = np.arange(total_time+tres, step=tres)

    X, T = np.meshgrid(x, t)

    u_t_0 = T0*np.sin(omega*t)
    u_t_1 = np.zeros(len(t))
    u_0_x = np.zeros(len(x))

    u = ftcs(sres, tres, u_t_0, u_t_1, u_0_x, k=1)

    fig = plt.figure()
    ax = plt.axes()
    c = ax.contourf(T, X, u, levels=int(np.max([3, total_space/sres+1])), cmap='plasma')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    fig.colorbar(c)
    fig.show()
    fig.savefig(f'heat_dynamic_{label}.png')

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(x, u[0], '.', label='t=0')
    ax.plot(x, u[int(np.around(0.25*T_osc / tres))], '.', label=r't=$\frac{1}{4}T$')
    ax.plot(x, u[int(np.around(0.5*T_osc / tres))], '.', label=r't=$\frac{1}{2}T$')
    ax.plot(x, u[int(np.around(0.75*T_osc / tres))], '.', label=r't=$\frac{3}{4}T$')
    ax.plot(x, u[int(np.around(1*T_osc / tres))], '.', label=r't=$T$')
    ax.plot(x, u[int(np.around(1.25 * T_osc / tres))], '.', label=r't=$1\frac{1}{4}T$')
    ax.plot(x, u[int(np.around(1.5 * T_osc / tres))], '.', label=r't=$1\frac{1}{2}T$')
    ax.plot(x, u[int(np.around(1.75 * T_osc / tres))], '.', label=r't=$1\frac{3}{4}T$')
    ax.plot(x, u[int(np.around(2 * T_osc / tres))], '.', label=r't=$2T$')
    ax.set_xlabel('x')
    ax.set_ylabel(r'$T\left(x,t\right)$')
    ax.legend(loc='upper left', bbox_to_anchor=(0.35, 1.0),
              ncol=3, fancybox=True)
    ax.grid()
    fig.show()
    fig.savefig(f'heat_dynamic_xplots_{label}.png')

# solve(2*np.pi*1e1, 5/10, 'fast')
solve(2*np.pi*1e-1, 30, 'slow')

