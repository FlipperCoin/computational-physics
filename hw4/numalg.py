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

    # for n in tqdm(range(0, 500)):
    for n in tqdm(range(0, a.shape[2]-1)):
        a[:, 0, n] = a[:, a.shape[1] - 2, n] if periodic else -a[:, 1, n]
        a[:, a.shape[1] - 1, n] = a[:, 1, n] if periodic else -a[:, a.shape[1] - 2, n]
        fn = f(a[:, :, n])
        for i in range(1, a.shape[1]-1):
            a[:, i, n+1] = (a[:, i+1, n] + a[:, i-1, n])/2 - tau/(2*h) * (fn[:, i+1] - fn[:, i-1])
            # !HOTFIX!
            if a[0, i, n + 1] <= 0:
                a[0, i, n + 1] = 0
                a[1, i, n + 1] = 0

    return a[:, 1:-1, :]

def gas_ftcs(rho0, v0, h, tau, time, periodic=True):
    xN = len(rho0)
    steps = int(time/tau)
    rho = np.zeros((steps+1, xN))
    rho[0, :] = rho0
    v = np.zeros((steps+1, xN))
    v[0, :] = v0

    for n in tqdm(range(0, steps)):
        for i in range(0, xN):
            rho[n+1, i] = (rho[n, (i+1)%xN] + rho[n, i-1])/2 - (tau/(2*h))*(rho[n, (i+1)%xN]*v[n, (i+1)%xN] - rho[n, i-1]*v[n, i-1])
            if np.abs(rho[n+1, i]) < 0.0001:
                v[n + 1, i] = 0
                continue
            v[n+1, i] = (1/rho[n+1, i]) * ((rho[n, (i+1)%xN]*v[n, (i+1)%xN] + rho[n, i-1]*v[n, i-1])/2 - (tau/(2*h))*(rho[n, (i+1)%xN]*(v[n, (i+1)%xN]**2) - rho[n, i-1]*(v[n, i-1]**2) + rho[n, (i+1)%xN]**(5/3) - rho[n, i-1]**(5/3)))

    return rho, v

def rho_order2(rho, v, h, tau, n, i):
    xN = rho.shape[1]
    rho_i_n = rho[n, i]
    order1 = -(tau/(2*h))*(rho[n, (i+1)%xN]*v[n, (i+1)%xN] - rho[n, i-1]*v[n, i-1])
    d2 = (tau**2) / (2*(h**2))
    order21 = d2*(rho[n, (i+1)%xN]*((v[n, (i+1)%xN])**2) + rho[n, (i-1)]*((v[n, (i-1)])**2) - 2*rho[n, i]*(v[n, i]**2))
    order22 = d2*(rho[n, (i+1)%xN]**(5/3) + rho[n, (i-1)]**(5/3) - 2*(rho[n, i]**(5/3)))
    order23 = 0 # no g
    order2 = order21 + order22 + order23
    rho_i_np1 = rho_i_n + order1 + order2

    return rho_i_np1

def v_order2(rho, v, h, tau, n, i):
    xN = rho.shape[1]
    v_i_n = v[n, i]
    order11 = ((v[n,  (i+1)%xN])**2 - (v[n, i-1])**2)
    order12 = (5/2)*((rho[n, (i+1)%xN])**(2/3)-(rho[n, i-1])**(2/3))
    order1 = (-tau/(2*h)) * (order11 + order12)
    v_ip1 = v[n,  (i+1) % xN]
    v_im1 = v[n, i-1]
    v_i = v[n, i]
    rho_ip1 = rho[n, (i+1) % xN]
    rho_im1 = rho[n, i-1]
    rho_i = rho[n, i]
    order21 = (v_ip1 + v_im1 - 2*v_i)*(v_i**2+(5/3)*(rho_i**(2/3)))
    order22 = 5*v_i*(rho_ip1**(2/3) + rho_im1**(2/3) - 2*(rho_i**(2/3)))
    order23 = (1/4)*(v_ip1 - v_im1)*(v_ip1**2 - v_im1**2 + (20/3)*(rho_ip1**(2/3) - rho_im1**(2/3)))
    order2 = (tau**2) / (2*(h**2)) * (order21 + order22 + order23)

    v_i_np1 = v_i_n + order1 + order2

    return v_i_np1

def gas_order2(rho0, v0, h, tau, time, periodic=True):
    xN = len(rho0)
    steps = int(time/tau)
    rho = np.zeros((steps+1, xN))
    rho[0, :] = rho0
    v = np.zeros((steps+1, xN))
    v[0, :] = v0

    for n in tqdm(range(0, steps)):
        for i in range(0, xN):
            rho[n+1, i] = rho_order2(rho, v, h, tau, n, i)
            if rho[n+1, i] <= 0:
                rho[n + 1, i] = 0
                v[n + 1, i] = 0
                continue
            v[n+1, i] = v_order2(rho, v, h, tau, n, i)

    return rho, v


def gas_order2_wall(rho0, v0, h, tau, time):
    xN = len(rho0)
    steps = int(time/tau)
    rho = np.zeros((steps+1, xN))
    rho[0, :] = rho0
    v = np.zeros((steps+1, xN))
    v[0, :] = v0

    def _rho(n, i):
        if i == -1:
            return rho[n, 0]
        if i == xN:
            return rho[n, xN-1]
        return rho[n, i]

    def _v(n, i):
        if i == -1:
            return -v[n, 0]
        if i == xN:
            return -v[n, xN-1]
        return v[n, i]

    # def rho_order2_wall(rho, v, h, tau, n, i):
    #     xN = rho.shape[1]
    #     rho_i_n = _rho(n, i)
    #     order1 = -(tau / (2 * h)) * (_rho(n, i+1) * _v(n, i+1) - _rho(n, i-1) * _v(n, i-1))
    #     d2 = (tau ** 2) / (2 * (h ** 2))
    #     order21 = d2 * (
    #                 _rho(n, i+1) * ((_v(n, i+1)) ** 2) + _rho(n, i-1) * (_v(n, i+1) ** 2) - 2 *
    #                 _rho(n, i) * (_v(n, i) ** 2))
    #     order22 = d2 * ((_rho(n, i+1) ** (5 / 3)) + (_rho(n, i-1) ** (5 / 3)) - 2 * (_rho(n, i) ** (5 / 3)))
    #     order23 = -(10/2)*d2*(_rho(n, i+1)-_rho(n, i-1))
    #     if i == 0:
    #         order1 += -(tau / (2 * h)) * (_rho(n, i)*_v(n, i))
    #         order22 += d2 * (_rho(n, i)*((_v(n, i))**2) + (_rho(n, i) ** (5/3)))
    #     elif i == (xN-1):
    #         order1 += -(tau / (2 * h)) * (-_rho(n, i)*_v(n, i))
    #         order22 += d2 * (_rho(n, i)*((_v(n, i))**2) + (_rho(n, i) ** (5/3)))
    #     order2 = order21 + order22 + order23
    #     rho_i_np1 = rho_i_n + order1 + order2
    #
    #     return rho_i_np1

    def rho_order2_wall(rho, v, h, tau, n, i):
        p = lambda t,x,k=1: (_rho(t,x))**(5/3)
        corrections = 0
        first_order = - tau / (2 * h) * (_rho(n, i + 1) * _v(n, i + 1) - _rho(n, i - 1) * _v(n, i - 1))
        second_order = tau ** 2 / (2 * h) * (1 / h * (
                    _rho(n, i + 1) * _v(n, i + 1) ** 2 + p(n, i + 1) + _rho(n, i - 1) * _v(n, i - 1) ** 2 + p(n,
                                                                                                              i - 1) - 2 * (
                                _rho(n, i) * _v(n, i) ** 2 + p(n, i))) - (-10) / 2 * (p(n, i + 1) - p(n, i - 1)))
        if i == 0:
            # corrections += -tau / (2 * h) * _rho(n, i) * _v(n, i)
            # corrections += tau ** 2 / (2 * h) * (1 / h * (_rho(n, i) * _v(n, i) ** 2) + p(n, i))
            corrections += -(-10) * ((tau ** 2) / (2 * h)) * _rho(n, i)

        if i == (xN - 1):
            corrections += -(-10) * ((tau ** 2) / (4 * h)) * _rho(n, i)
            # corrections += tau / (2 * h) * _rho(n, i) * _v(n, i)
            # corrections += tau ** 2 / (2 * h) * (1 / h * (_rho(n, i) * _v(n, i) ** 2) + p(n, i))


        return _rho(n, i) + first_order + second_order + corrections
    #
    # def v_order2_wall(rho, v, h, tau, n, i):
    #         xN = rho.shape[1]
    #         v_i_n = _v(n, i)
    #         order11 = ((_v(n, i+1)) ** 2 - (_v(n, i-1)) ** 2 - 2*h*(-10))
    #         order12 = (5 / 2) * ((_rho(n, i+1)) ** (2 / 3) - (_rho(n, i - 1)) ** (2 / 3))
    #         order1 = (-tau / (2 * h)) * (order11 + order12)
    #         v_ip1 = _v(n, i + 1)
    #         v_im1 =  _v(n, i - 1)
    #         v_i = _v(n, i)
    #         rho_ip1 = _rho(n, i + 1)
    #         rho_im1 = _rho(n, i - 1)
    #         rho_i = _rho(n, i)
    #         order21 = (v_ip1 + v_im1 - 2 * v_i) * (v_i ** 2 + (5 / 3) * (rho_i ** (2 / 3)))
    #         order22 = 5 * v_i * (rho_ip1 ** (2 / 3) + rho_im1 ** (2 / 3) - 2 * (rho_i ** (2 / 3)))
    #         order23 = (1 / 4) * (v_ip1 - v_im1) * (
    #                     v_ip1 ** 2 - v_im1 ** 2 + (20 / 3) * (rho_ip1 ** (2 / 3) - rho_im1 ** (2 / 3)))
    #         order24 = (1/4) * (v_ip1 - v_im1) * (-2 * h * (-10))
    #         order2 = (tau ** 2) / (2 * (h ** 2)) * (order21 + order22 + order23 + order24)
    #
    #         v_i_np1 = v_i_n + order1 + order2
    #
    #         return v_i_np1
    def v_order2_wall(rho, v, h, tau, n, i):
        first_order = tau * (-10) - tau / (2 * h) * (1 / 2 * (_v(n, i + 1) ** 2 - _v(n, i - 1) ** 2) + 5 / 2 * (
                    _rho(n, i + 1) ** (2 / 3) - _rho(n, i - 1) ** (2 / 3)))
        second_order1 = (tau ** 2) / (8 * h ** 2) * ((_v(n, i + 1) ** 2 - _v(n, i - 1) ** 2 + 20 / 3 * (
                    _rho(n, i + 1) ** (2 / 3) - _rho(n, i - 1) ** (2 / 3)) - 2 * h * (-10)) * (_v(n, i + 1) - _v(n, i - 1)))
        second_order2 = (tau ** 2) / (2 * h ** 2) * ((_v(n, i) ** 2 + 5 / 3 * _rho(n, i) ** (2 / 3)) * (
                    _v(n, i + 1) + _v(n, i - 1) - 2 * _v(n, i)) + 5 * (_rho(n, i + 1) ** (2 / 3) + _rho(n, i - 1) ** (
                    2 / 3) - 2 * _rho(n, i) ** (2 / 3)) * _v(n, i))
        return _v(n, i) + first_order + second_order1 + second_order2

    for n in tqdm(range(0, steps)):
        for i in range(0, xN):
            rho[n+1, i] = rho_order2_wall(rho, v, h, tau, n, i)
            if rho[n+1, i] <= 0:
                rho[n + 1, i] = 0
                v[n + 1, i] = 0
                continue
            v[n+1, i] = v_order2_wall(rho, v, h, tau, n, i)

    return rho, v
