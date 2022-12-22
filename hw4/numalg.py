import numpy as np
from tqdm import tqdm

def euler_step(x, f, tau):
    return x + tau * f(x)

def euler_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps+1, d))
    x[0] = x0

    for n in range(0, steps):
        x[n+1] = euler_step(x[n], f, tau)

    return x

def rk2_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps + 1, d))
    x[0] = x0

    for n in range(0, steps):
        x[n + 1] = x[n] + tau * f(euler_step(x[n], f, tau/2))

    return x

def rk4_method(T, steps, f, x0):
    tau = T / steps
    d = len(x0)
    x = np.zeros((steps + 1, d))
    x[0] = x0

    for n in range(0, steps):
        F1 = f(x[n])
        F2 = f(x[n] + (1/2)*tau*F1)
        F3 = f(x[n] + (1/2)*tau*F2)
        F4 = f(x[n] + tau*F3)
        x[n + 1] = x[n] + (1/6)*tau*(F1+2*F2+2*F3+F4)

    return x

def rk4_adaptive_method(T, f, x0, delta, eps=0.01, init_tau=None, s1=0.9, s2=2, max_attempts=100):
    tau = init_tau if init_tau else T / 10000
    d = len(x0)
    x = np.zeros((1, d))
    x[0] = x0

    def step(xn, t, tau):
        F1 = f(xn, t)
        F2 = f(xn + (1 / 2) * tau * F1, t + tau/2)
        F3 = f(xn + (1 / 2) * tau * F2, t + tau/2)
        F4 = f(xn + tau * F3, t + tau)
        return xn + (1 / 6) * tau * (F1 + 2 * F2 + 2 * F3 + F4)

    def adaptive_step(xn, t, tau):
        for i in range(0, max_attempts):
            big_step = step(xn, t, tau)
            small_step = step(step(xn, t, tau/2), t+tau/2, tau/2)

            current_delta = delta(big_step, small_step, t+tau)
            tau_est = tau*np.power(eps/current_delta, 1/5)
            tau_est *= s1
            if tau_est > s2*tau:
                tau_est = s2*tau
            elif tau_est < tau/s2:
                tau_est = tau/s2

            tau_old = tau
            tau = tau_est

            if current_delta < eps:
                return big_step, t+tau_old, tau

        raise Exception('reached max attempts to satisfy eps.')

    t = np.array([0])
    while t[-1] < T:
        if t[-1]+tau <= T:
            x_new, t_new, tau = adaptive_step(x[-1], t[-1], tau)
        else:
            x_new, t_new, tau = adaptive_step(x[-1], t[-1], T-t[-1])

        x = np.vstack((x, x_new))
        t = np.append(t, t_new)

    return x, t

def heat_ftcs(sres, tres, bc0, bc1, ic0, k=1):
    tf = len(bc0)
    xf = len(ic0)
    u = np.zeros((tf, xf))
    u[0] = ic0
    u[:, 0] = bc0
    u[:, xf-1] = bc1

    for t in tqdm(range(0, tf-1)):
        for x in range(1, xf-1):
            u[t+1, x] = u[t, x] + (k*tres/(sres**2))*(u[t, x+1] + u[t, x-1] - 2*u[t, x])

    return u

def advection_ftcs(f, source, ic, h, tau, time, periodic=True):
    """

    Args:
        f: flux
        source: source
        ic: initial conditions
        h: space resolution
        tau: time resolution
        time: total time to compute
        periodic: periodic boundary conditions if True, else wall boundary conditions

    Returns:

    """

    d = ic.shape
    d = (d[0], d[1] + 2)
    steps = int(time / tau)
    a = np.zeros(d + (steps+1,))
    a[:, 1:-1, 0] = ic

    for n in tqdm(range(0, a.shape[2]-1)):
        a[:, 0, n] = a[:, a.shape[1] - 2, n] if periodic else -a[:, 1, n]
        a[:, a.shape[1] - 1, n] = a[:, 1, n] if periodic else -a[:, a.shape[1] - 2, n]
        fn = f(a[:, :, n])
        for i in range(1, a.shape[1]-1):
            a[:, i, n+1] = a[:, i, n] - tau/(2*h) * (fn[:, i+1] - fn[:, i-1])
            # !HOTFIX!
            if a[0, i, n + 1] < 0:
                a[0, i, n + 1] = 0

    return a[:, 1:-1, :]

