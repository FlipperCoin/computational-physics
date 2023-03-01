import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import cumtrapz

from numalg import gas_order2

L = 1
rho0 = 1
# speed of sound
cs0 = np.sqrt(5/3 * rho0**(2/3))
ts = L/cs0
T = 4*ts
tau = 1e-3 * ts
h = 1e-2

x_num = int(L/h) + 1
xn = np.arange(0, L+h, h)
tn = np.arange(0, T+tau, tau)

rho0 = np.zeros(x_num)
rho0[int(0.9/h):] = 1
v0 = np.zeros(x_num)

rho_analytic = ((2/5)*(-10)*xn + 1.236)**(3/2)
rho_analytic[xn > (5/20*1.236)] = 0

rho_o2, v_o2 = gas_order2(rho0, v0, h, tau, T)

fig, axs = plt.subplots()
# tN = [0, 1000, 2500, 5000, 10000]
tN = [0, 250, 400, 1000, 2000, 4000]
for ti in tN:
    axs.plot(xn, rho_o2[ti, :], label=f't={tn[ti]:.3f}')
axs.plot(xn, rho_analytic, label=f'analytic')
axs.set_xlabel('x')
axs.set_ylabel(r'$\rho$')
axs.grid()
axs.legend()
fig.savefig('gravity_density_evolution.png')
fig.show()

total_mass_o2 = np.sum(rho_o2, 1)*h

fig, axs = plt.subplots()
axs.plot(tn, total_mass_o2)
axs.set_xlabel('t')
axs.set_ylabel('total mass')
axs.grid()
fig.savefig('gravity_mass.png')
fig.show()

total_momentum_o2 = np.sum(rho_o2*v_o2, 1)*h
# total_momentum[np.abs(total_momentum - 0) < 1e-8] = 0

fig, axs = plt.subplots()
axs.plot(tn, total_momentum_o2)
axs.set_xlabel('t')
axs.set_ylabel('total momentum')
axs.grid()
fig.savefig('gravity_momentum.png')
fig.show()

print('done.')
