import numpy as np
import matplotlib.pyplot as plt

from numalg import relaxed_poisson

#%% q2
h = 1e-1
x = np.arange(-np.sqrt(10), np.sqrt(10), h)
y = np.arange(-np.sqrt(10), np.sqrt(10), h)
z = np.arange(-np.sqrt(10), np.sqrt(10), h)

X, Y, Z = np.meshgrid(x, y, z)

def index(cor, axis):
    return int((cor + np.sqrt(10)) / (2*np.sqrt(10)) * np.size(axis))

def i(cor):
    return index(cor, x)

def j(cor):
    return index(cor, y)

def k(cor):
    return index(cor, z)

rho = np.zeros(X.shape)

phi0_mat = np.zeros(X.shape)

box_r = 1.5
phi_s = 1
phi0_mat[i(box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r):j(box_r),k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r):j(box_r),k(-box_r)] = phi_s

ground_cor = np.abs(X**2 + Y**2 + Z**2 - 10) < h
phi0_mat[ground_cor] = 0

mask = np.zeros(phi0_mat.shape)
mask[phi0_mat != 0] = 1
mask[ground_cor] = 1
phi0_args = (phi0_mat, mask)

(phi_sol, n) = relaxed_poisson(phi0_args, rho, h, (1/5)*1e-2)



plt.imshow(phi_sol[34,:,:])
plt.colorbar()
plt.show()
