import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import time

from numalg import rk4_adaptive_method

def system(t):
    x1 = np.sin(t)
    y1 = np.cos(t)
    x2 = -np.sin(t)
    y2 = -np.cos(t)

    return np.array([[x1, y1], [x2, y2]])

def two_body_force(x1, y1, x2, y2):
    delx = x2-x1
    dely = y2-y1
    r_cubed = (delx ** 2 + dely ** 2) ** (3/2)
    force_x = - delx / r_cubed
    force_y = - dely / r_cubed

    return np.array([force_x, force_y])

def two_body_potential(x1, y1, x2, y2):
    delx = x2 - x1
    dely = y2 - y1
    r = (delx ** 2 + dely ** 2) ** (1/2)

    return -(1/r)

def partial_three_body_f(x, t):
    bodies = system(t)

    force = two_body_force(bodies[0][0], bodies[0][1], x[0], x[1]) \
            + two_body_force(bodies[1][0], bodies[1][1], x[0], x[1])

    fx = x[2]
    fy = x[3]
    fvx = force[0]
    fvy = force[1]
    return np.array([fx, fy, fvx, fvy])

def particle_energy(x, t):
    kinetic = 1/2 * (x[2]**2 + x[3]**2)

    bodies = system(t)
    potential = two_body_potential(bodies[0][0], bodies[0][1], x[0], x[1]) \
                + two_body_potential(bodies[1][0], bodies[1][1], x[0], x[1])

    return kinetic + potential

def rel_error(est, real):
    return np.abs((est - real) / real)

def delta(big_step, small_step, t):
    e_small = particle_energy(small_step, t)
    e_big = particle_energy(big_step, t)

    return rel_error(e_big, e_small)

x0 = np.array([1.5, 0, -1, 0])
T = 6*np.pi

st = time()
eps = 0.00001
x, t = rk4_adaptive_method(T, partial_three_body_f, x0, delta, eps=eps)
ed = time()
runtime = ed - st

print(f'rk4 adaptive result for T={T}, eps={eps}. runtime {runtime} sec, {len(t)-1} steps')

e0 = particle_energy(x0, 0)
e = particle_energy(np.transpose(x), t)

bodies = system(t)
bodies_polar = np.array([[np.arctan2(bodies[0][1], bodies[0][0]), np.sqrt(bodies[0][0] ** 2 + bodies[0][1] ** 2)],
                         [np.arctan2(bodies[1][1], bodies[1][0]), np.sqrt(bodies[1][0] ** 2 + bodies[1][1] ** 2)]])

x_fig = plt.figure(3)
x_axs = x_fig.subplots()
y_fig = plt.figure(4)
y_axs = y_fig.subplots()
polar_fig = plt.figure(5)
polar_axs = polar_fig.subplots(subplot_kw={'projection': 'polar'})
e_fig = plt.figure(6)
e_axs = e_fig.subplots()

x_axs.plot(t, x[:, 0], '.')
y_axs.plot(t, x[:, 1], '.')
r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
theta = np.arctan2(x[:, 1], x[:, 0])
polar_axs.plot(theta, r, '.')
e_axs.plot(t, rel_error(e, e0), '.')

x_axs.set_xlabel(r't')
x_axs.set_ylabel(r'x')
x_axs.grid()
x_fig.savefig('three_body_adaptive_xt.png')
x_fig.show()

y_axs.set_xlabel(r't')
y_axs.set_ylabel(r'y')
y_axs.grid()
y_fig.savefig('three_body_adaptive_yt.png')
y_fig.show()

polar_axs.grid(True)
# polar_axs.set_rticks([0.5, 1, 1.5])
polar_fig.savefig('three_body_adaptive_polar.png')
polar_fig.show()

e_axs.set_xlabel(r't')
e_axs.set_ylabel(r'$\epsilon = \frac{\Delta E}{E}$')
e_axs.grid()
e_fig.savefig('three_body_adaptive_error.png')
e_fig.show()

def create_animate(axs, t, x, y, frames):
    def animate(i):
        x[-1] / frames
        axs.clear()
        axs.grid(True)
        axs.plot(x[i], y[i], '.', label='particle')
        axs.plot(bodies_polar[0][0][i], bodies_polar[0][1][i], '.', label='body 1')
        axs.plot(bodies_polar[1][0][i], bodies_polar[1][1][i], '.', label='body 2')
        axs.set_rticks([1, 2, 3, 4, 5, 6])
        axs.legend()

    return animate

polar_ani_fig = plt.figure()
polar_ani_axs = polar_ani_fig.subplots(subplot_kw={'projection': 'polar'})
frames = 1000
polar_ani = FuncAnimation(polar_ani_fig, create_animate(polar_ani_axs, t, theta, r, frames), frames=frames, interval=50, repeat=False)
polar_ani.save('three_body_adaptive_xt.mp4')
print('done.')
