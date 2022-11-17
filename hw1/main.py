import numpy as np
import matplotlib.pyplot as plt
from math import factorial as fact
import sys
from scipy.constants import g

print("======= 1 =======")
print("")

min_system = sys.float_info.min
max_system = sys.float_info.max

print(f"sys.float_info.min value: {sys.float_info.min}")
print(f"sys.float_info.max value: {sys.float_info.max}")

min_search = 1
while min_search/10 != 0:
    min_search/=10

max_search = float(1)
while max_search*10 != float('inf'):
    max_search*=10

print(f"searched min value: {min_search}")
print(f"searched max value: {max_search}")

min = min_search
max = 1e308

eps = 1
while 1 + eps != 1:
    eps/=10

print(f"eps value: {eps}")

print(f"sqrt(1.1) value: {np.sqrt(1.1)}")


def sqrt_taylor_term(n=0, x=0.1):
    """evaluate the nth term in the taylor series of sqrt(1+x).

    Keyword arguments:
    n -- term number starting from 0
    x -- the value to place in sqrt(1+x)
    """
    return 1/(fact(n)) * np.multiply.reduce(np.arange(1/2,1/2-n,-1)) * x**n

n = 0
sqrt_fine = np.sqrt(1.1)
while np.sum([sqrt_taylor_term(k) for k in np.arange(0,n+1)]) != sqrt_fine:
    n += 1

print(f"taylor series value (should be the same as sqrt(1.1)): {np.sum([sqrt_taylor_term(n) for n in np.arange(0,n+1)])}")
print(f"number of terms used: {n+1}")

print("")
print("======= 2 =======")
print("")

x0 = 5/6
def f(x): return np.sqrt(1+x)

print("f(x) = sqrt(1+x)")
print("df/dx = 1/2 * (1/sqrt(1+x)), x = 5/6")

def df(x): return 1/2*(1/np.sqrt(1+x))

def deltaf(x, deltax): return (f(x + deltax) - f(x)) / deltax
def R(x, deltax): return np.abs(((df(x) - deltaf(x,deltax)) / df(x)))

print(f"for deltax 1e-2, R: {R(x0, 1e-2)}")

deltax = np.logspace(-11, -5, 100)
y = R(x0, deltax)

plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta x$')
plt.ylabel('$R$')
plt.grid()
plt.plot(deltax, y, '.', label='R')
plt.savefig('2.png')
plt.show()

min_idx = np.argmin(y)
print(f"most accurate result - deltax: {deltax[min_idx]}, R: {y[min_idx]}")

print("")
print("======= 3 =======")
print("")

y0 = 3
x0 = 0
vx0 = 10
vy0 = 10
T = 2.3
t = np.linspace(0, T, 1000)

analytic_y = y0+vy0*t-1/2*g*t**2
analytic_x = x0+vx0*t

def euler_method(T, steps, a, v0, r0):
    tau = T / steps
    v = np.zeros(steps+1)
    r = np.zeros(steps+1)
    v[0] = v0
    r[0] = r0

    for n in np.arange(0, steps):
        v[n+1] = v[n] + tau*a(v[n], r[n])
        r[n+1] = r[n] + tau*v[n]

    return r

def ax(v, r): return 0
def ay(v, r): return -g
num_euler_10_x = euler_method(T, 10, ax, vx0, x0)
num_euler_10_y = euler_method(T, 10, ay, vy0, y0)
num_euler_100_x = euler_method(T, 100, ax, vx0, x0)
num_euler_100_y = euler_method(T, 100, ay, vy0, y0)

analytic_filter = analytic_y > 0
analytic_y = analytic_y[analytic_filter]
analytic_x = analytic_x[analytic_filter]
plt.ylabel(r'y $\left[ m \right]$')
plt.xlabel(r'x $\left[ m \right]$')
plt.plot(analytic_x, analytic_y, label='analytic')
plt.plot(num_euler_10_x, num_euler_10_y, '.', label='euler\'s method, 10 steps')
plt.plot(num_euler_100_x, num_euler_100_y, '.', label='euler\'s method, 100 steps')
plt.legend()
plt.grid()
plt.savefig('3_euler.png')
plt.show()

def midpoint_method(T, steps, a, v0, r0):
    tau = T / steps
    v = np.zeros(steps+1)
    r = np.zeros(steps+1)
    v[0] = v0
    r[0] = r0

    for n in np.arange(0, steps):
        v[n+1] = v[n] + tau*a(v[n], r[n])
        r[n+1] = r[n] + tau*(v[n+1] +v[n])/2

    return r

num_midpoint_25_x = midpoint_method(T, 25, ax, vx0, x0)
num_midpoint_25_y = midpoint_method(T, 25, ay, vy0, y0)
num_midpoint_100_x = midpoint_method(T, 100, ax, vx0, x0)
num_midpoint_100_y = midpoint_method(T, 100, ay, vy0, y0)
num_midpoint_5_x = midpoint_method(T, 5, ax, vx0, x0)
num_midpoint_5_y = midpoint_method(T, 5, ay, vy0, y0)
num_midpoint_1_x = midpoint_method(T, 1, ax, vx0, x0)
num_midpoint_1_y = midpoint_method(T, 1, ay, vy0, y0)

plt.ylabel(r'y $\left[ m \right]$')
plt.xlabel(r'x $\left[ m \right]$')
plt.plot(analytic_x, analytic_y, label='analytic')
plt.plot(num_midpoint_25_x, num_midpoint_25_y, '.', label='midpoint method, 25 steps')
plt.plot(num_midpoint_100_x, num_midpoint_100_y, '.', label='midpoint method, 100 steps')
plt.plot(num_midpoint_5_x, num_midpoint_5_y, '.', label='midpoint method, 5 steps')
plt.plot(num_midpoint_1_x, num_midpoint_1_y, '.', label='midpoint method, 1 steps')
plt.legend()
plt.grid()
plt.savefig('3_midpoint.png')
plt.show()
