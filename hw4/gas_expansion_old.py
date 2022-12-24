import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumtrapz

from numalg import advection_ftcs

def mass_f(a):
    # rho*v
    return a[1]

def momentum_f(a):
    # rho*v**2 + rho**(5/3)
    mom = a[1]
    mom_f = (mom**2 / a[0]) + a[0]**(5/3)
    cond = np.abs(a[0] - 0) < 0.001
    mom_f[cond] = 0

    return mom_f

def f(a):
    return np.array([mass_f(a), momentum_f(a)])

def acc_source(a, g):
    return a[0]*g

def no_g_source(a):
    return acc_source(a, 0)

L = 1
rho0 = 1
# speed of sound
cs0 = np.sqrt(5/3 * rho0**(2/3))
ts = L/cs0
T = 3*ts
tau = 1e-4 * ts
h = 1e-2

x_num = int(L/h) + 1
xn = np.arange(0, L+h, h)
tn = np.arange(0, T+tau, tau)

ic = np.zeros((2, x_num))
ic[0][int(0.9/h):] = 1

a = advection_ftcs(f, no_g_source, ic, h, tau, T, periodic=True)

total_mass = cumtrapz(a[0, :, :], xn, axis=0)[-1, :]

a_total = a
xn_total = xn
tn_total = tn
#
# cut = np.argmax(np.abs(total_mass - total_mass[0]) > 0.05)
#
# a = a_total[:, :, :cut]
# tn = tn[:cut]

fig, axs = plt.subplots()
axs.plot(tn, total_mass)
axs.set_xlabel('t')
axs.set_ylabel('total mass')
axs.grid()
fig.savefig('ftcs_mass.png')
fig.show()

fig, axs = plt.subplots()
tN = [0, 20, 50, 100, 500, 2000]
for ti in tN:
    axs.plot(xn, a[0, :, ti], label=f't={tn[ti]:.3f}')

# axs.plot(xn, a[0, :, 500], label=f't={tn[500]}')
axs.set_xlabel('x')
axs.set_ylabel(r'$\rho$')
axs.grid()
axs.legend()
fig.savefig('ftcs_density_evolution.png')
fig.show()

#
# def create_animate(axs, a):
#     def animate(i):
#         axs.clear()
#         axs.plot(xn, a[0, :, i], '.')
#         axs.grid()
#         axs.set_xlabel(r'$x$')
#         axs.set_ylabel(r'$\rho$')
#         axs.set_xlim([0, L])
#         # axs.set_ylim([0, 1.5])
#
#     return animate
#

# fig, ax = plt.subplots()
# animation = FuncAnimation(fig, create_animate(ax, a), frames=x_num, interval=100, repeat=False)
# animation.save(f'gas_expansion_ftcs.mp4')
# print('done.')

