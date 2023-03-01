import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumtrapz

from numalg import gas_ftcs, gas_order2, gas_order2_old

L = 1
rho0 = 1
# speed of sound
cs0 = np.sqrt(5/3 * rho0**(2/3))
ts = L/cs0
T = 6*ts
tau = 1e-3 * ts
h = 1e-2

x_num = int(L/h) + 1
xn = np.arange(0, L+h, h)
tn = np.arange(0, T+tau, tau)

rho0 = np.zeros(x_num)
rho0[int(0.9/h):] = 1
v0 = np.zeros(x_num)
#
# rho_o1, v_o1 = gas_ftcs(rho0, v0, h, tau, T)
#
# # total_mass = cumtrapz(rho[:, :], xn, axis=1)[:, -1]
# total_mass_o1 = np.sum(rho_o1, 1)*h
# total_mass_o1[np.abs(total_mass_o1 - 0.11) < 1e-8] = 0.11
#
# fig, axs = plt.subplots()
# axs.plot(tn, total_mass_o1)
# axs.set_xlabel('t')
# axs.set_ylabel('total mass')
# axs.grid()
# fig.savefig('ftcs_mass.png')
# fig.show()
#
# total_momentum_o1 = np.sum(rho_o1*v_o1, 1)*h
# # total_momentum[np.abs(total_momentum - 0) < 1e-8] = 0
#
# fig, axs = plt.subplots()
# axs.plot(tn, total_momentum_o1)
# axs.set_xlabel('t')
# axs.set_ylabel('total momentum')
# axs.grid()
# fig.savefig('ftcs_momentum.png')
# fig.show()
#
# fig, axs = plt.subplots()
# tN = [0, 20, 50, 100, 500, 2000, 30000]
# for ti in tN:
#     axs.plot(xn, rho_o1[ti, :], label=f't={tn[ti]:.3f}')
# axs.set_xlabel('x')
# axs.set_ylabel(r'$\rho$')
# axs.grid()
# axs.legend()
# fig.savefig('ftcs_density_evolution.png')
# fig.show()

# rho_o2, v_o2 = gas_order2(rho0, v0, h, tau, T, g=0, periodic=True)
rho_o2, v_o2 = gas_order2_old(rho0, v0, h, tau, T, periodic=True)

fig, axs = plt.subplots()
# tN = [0, 1000, 2500, 5000, 15000, 30000]
tN = [0, 250, 500, 830, 6000]
for ti in tN:
    axs.plot(xn, rho_o2[ti, :], label=f't={tn[ti]:.3f}')
axs.set_xlabel('x')
axs.set_ylabel(r'$\rho$')
axs.grid()
axs.legend()
# fig.savefig('order2_density_evolution.png')
fig.show()

total_mass_o2 = np.sum(rho_o2, 1)*h
total_mass_o2[np.abs(total_mass_o2 - 0.11) < 1e-8] = 0.11

fig, axs = plt.subplots()
axs.plot(tn, total_mass_o2)
axs.set_xlabel('t')
axs.set_ylabel('total mass')
axs.grid()
# fig.savefig('order2_mass.png')
fig.show()

total_momentum_o2 = np.sum(rho_o2*v_o2, 1)*h
# total_momentum[np.abs(total_momentum - 0) < 1e-8] = 0

fig, axs = plt.subplots()
axs.plot(tn, total_momentum_o2)
axs.set_xlabel('t')
axs.set_ylabel('total momentum')
axs.grid()
# fig.savefig('order2_momentum.png')
fig.show()

print('done.')

