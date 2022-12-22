import numpy as np
import matplotlib.pyplot as plt
from time import time

from numalg import ftcs

T0 = 1

total_time = 1
total_space = 1
tres = 0.00001
sres = 0.01

x = np.arange(total_space+sres, step=sres)
t = np.arange(total_time+tres, step=tres)

X, T = np.meshgrid(x, t)

u_t_0 = T0*np.ones(len(t))
u_t_1 = np.zeros(len(t))
u_0_x = np.zeros(len(x))

u = ftcs(sres, tres, u_t_0, u_t_1, u_0_x, k=1)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter(T, X, u, c=u, cmap='plasma')
# ax.view_init(30, 40+90)
# fig.show()
# fig.savefig('heat3D.png')

fig = plt.figure()
ax = plt.axes()
c = ax.contourf(T, X, u, levels=int(np.max([3, total_space/sres+1])), cmap='plasma')
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.colorbar(c)
fig.show()
fig.savefig('heat.png')

fig = plt.figure()
ax = plt.axes()
ax.plot(x, u[0], '.', label='t=0')
ax.plot(x, u[int(np.around(0.01/tres))], '.', label='t=0.01')
ax.plot(x, u[int(np.around(0.02/tres))], '.', label='t=0.02')
ax.plot(x, u[int(np.around(0.04/tres))], '.', label='t=0.04')
ax.plot(x, u[int(np.around(0.08/tres))], '.', label='t=0.08')
ax.plot(x, u[int(np.around(0.16/tres))], '.', label='t=0.16')
ax.plot(x, u[int(np.around(0.3/tres))], '.', label='t=0.30')
ax.plot(x, u[int(np.around(1/tres))], '.', label='t=1')
ax.set_xlabel('x')
ax.set_ylabel(r'$T\left(x,t\right)$')
ax.legend()
ax.grid()
fig.show()
fig.savefig('heat_xplots.png')

