import numpy as np
import matplotlib.pyplot as plt
from time import time

from numalg import ftcs

T0 = 1
omega = 1

total_time = 5*2*np.pi
total_space = 1
tres = 0.001
sres = 0.1

x = np.arange(total_space+sres, step=sres)
t = np.arange(total_time+tres, step=tres)

X, T = np.meshgrid(x, t)

u_t_0 = T0*np.sin(omega*t)
u_t_1 = np.zeros(len(t))
u_0_x = np.zeros(len(x))

u = ftcs(sres, tres, u_t_0, u_t_1, u_0_x, k=1)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(T, X, u, c=u, cmap='plasma')
ax.view_init(30, 40+90)
fig.show()
fig.savefig('heat3D_dynamic.png')

fig = plt.figure()
ax = plt.axes()
ax.contourf(T, X, u, levels=int(np.max([3, total_space/sres+1])), cmap='plasma')
ax.set_xlabel('t')
ax.set_ylabel('x')
fig.show()
fig.savefig('heat_dynamic.png')

