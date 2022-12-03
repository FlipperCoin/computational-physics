import numpy as np
import matplotlib.pyplot as plt
from time import time

from numalg import ftcs

T0 = 1

total_time = 50
total_space = 1
tres = 0.01
sres = 0.01

x = np.arange(total_space+sres, step=sres)
t = np.arange(total_time+tres, step=tres)

X, T = np.meshgrid(x, t)

u_t_0 = T0*np.ones(len(t))
u_t_1 = np.zeros(len(t))
u_0_x = np.zeros(len(x))

u = ftcs(sres, tres, u_t_0, u_t_1, u_0_x)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(T, X, u, cmap='plasma')
ax.view_init(30, 40+90)
fig.show()
fig.savefig('heat3D.png')

fig = plt.figure()
ax = plt.axes()
ax.contourf(T, X, u, levels=25)
fig.show()
fig.savefig('heat.png')



