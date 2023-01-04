import numpy as np
import matplotlib.pyplot as plt
from time import time

from numalg import relaxed_poisson

h = 1e-1
box_r = 1.5
x = np.arange(-np.sqrt(10), np.sqrt(10), h)
y = np.arange(-np.sqrt(10), np.sqrt(10), h)
z = np.arange(-np.sqrt(10), np.sqrt(10), h)

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
ground_cor = np.abs(X**2 + Y**2 + Z**2 - 10) < h

def index(cor, axis):
    return int((cor + np.sqrt(10)) / (2*np.sqrt(10)) * np.size(axis))

def i(cor):
    return index(cor, x)

def j(cor):
    return index(cor, y)

def k(cor):
    return index(cor, z)

def fill_reflection_f(f):
    i0 = f.shape[0] // 2
    j0 = f.shape[1] // 2
    k0 = f.shape[2] // 2

    f[:i0, j0:, k0:, :] = np.flip(f, 0)[:i0, j0:, k0:, :]
    f[:i0, j0:, k0:, 0] *= -1
    f[:, j0:, :k0, :] = np.flip(f, 2)[:, j0:, :k0, :]
    f[:, j0:, :k0, 2] *= -1
    f[:, :j0, :, :] = np.flip(f, 1)[:, :j0, :, :]
    f[:, :j0, :, 1] *= -1

    return f

def fill_reflection_rho(f):
    i0 = f.shape[0] // 2
    j0 = f.shape[1] // 2
    k0 = f.shape[2] // 2

    f[:i0, j0:, k0:] = np.flip(f, 0)[:i0, j0:, k0:]
    f[:, j0:, :k0] = np.flip(f, 2)[:, j0:, :k0]
    f[:, :j0, :] = np.flip(f, 1)[:, :j0, :]

    return f

def e_field(phi):
    f = np.zeros(phi.shape + (3,))

    i0 = phi.shape[0]//2
    j0 = phi.shape[1]//2
    k0 = phi.shape[2]//2

    for i in range(i0,phi.shape[0]-1):
        for j in range(j0,phi.shape[1]-1):
            for k in range(k0,phi.shape[2]-1):
                f[i,j,k,0] = (-1/(2*h))*(phi[i+1,j,k] - phi[i-1,j,k])
                f[i,j,k,1] = (-1/(2*h))*(phi[i,j+1,k] - phi[i,j-1,k])
                f[i,j,k,2] = (-1/(2*h))*(phi[i,j,k+1] - phi[i,j,k-1])

    f = fill_reflection_f(f)
    return f

def rho(phi):
    rho = np.zeros(phi.shape)

    i0 = phi.shape[0]//2
    j0 = phi.shape[1]//2
    k0 = phi.shape[2]//2

    for i in range(i0,phi.shape[0]-1):
        for j in range(j0,phi.shape[1]-1):
            for k in range(k0,phi.shape[2]-1):
                rho[i,j,k] = (-1/(4*np.pi*(h**2)))*(phi[i+1,j,k] + phi[i-1,j,k] + phi[i,j+1,k] + phi[i,j-1,k] + phi[i,j,k+1] + phi[i,j,k-1] - 6*phi[i,j,k])
                # rho[i,j,k] = (-1/(4*np.pi*(h**2)))*(rho[i+1,j,k]+rho[i-1,j,k]+rho[i,j+1,k]+phi[i,j-1,k]+rho[i,j,k+1]+rho[i,j,k-1]) \
                #                       + (-1/(4*np.pi*(h**2)))*(rho[i+1,j+1,k]+rho[i+1,j-1,k]+rho[i+1,j,k+1]+rho[i+1,j,k-1]+rho[i+1,j+1,k+1]+rho[i+1,j+1,k-1]+rho[i+1,j-1,k+1]+rho[i+1,j-1,k-1]) + \
                #                       (-1/(4*np.pi*(h**2)))*(rho[i-1,j+1,k]+rho[i-1,j-1,k]+rho[i-1,j,k+1]+rho[i-1,j,k-1]+rho[i-1,j+1,k+1]+rho[i-1,j+1,k-1]+rho[i-1,j-1,k+1]+rho[i-1,j-1,k-1]) + \
                #                       (-1/(4*np.pi*(h**2)))*(rho[i,j+1,k+1]+rho[i,j+1,k-1]+rho[i,j-1,k+1]+rho[i,j-1,k-1])

    rho = fill_reflection_rho(rho)
    return rho


#%% q2

rho0 = np.zeros(X.shape)

phi0_mat = np.zeros(X.shape)

phi_s = 1
phi0_mat[i(box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r):j(box_r),k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r):j(box_r),k(-box_r)] = phi_s

phi0_mat[ground_cor] = 0

mask = np.zeros(phi0_mat.shape)
mask[phi0_mat != 0] = 1
mask[ground_cor] = 1
phi0_args = (phi0_mat, mask)

print("q2.")
(q2_phi_sol, n, sec_elapsed) = relaxed_poisson(phi0_args, rho0, h, 1e-3)

print(f"phi complete; n: {n}, total time: {sec_elapsed} sec.")

start = time()
q2_f = e_field(q2_phi_sol)
stop = time()

print(f"E complete; total time: {stop-start} sec.")

start = time()
q2_rho_sol = rho(q2_phi_sol)
stop = time()

print(f"rho complete; total time: {stop-start} sec.")

plt.figure(dpi=150)
plt.imshow(q2_phi_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\Phi$")
plt.savefig("q2_phi_inside.png")
plt.show()

plt.figure(dpi=150)
plt.quiver(Y[i(0),:,:], Z[i(0),:,:], q2_f[i(0),:,:,1], q2_f[i(0),:,:,2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("q2_E_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q2_rho_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q2_rho_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q2_rho_sol[i(-box_r),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q2_rho_surface.png")
plt.show()

#%% q3

rho0 = np.zeros(X.shape)

phi0_mat = 1/np.sqrt(X**2 + Y**2 + Z**2)
mask = np.zeros(phi0_mat.shape)

mask[i(box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = 1
mask[i(-box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = 1
mask[i(-box_r):i(box_r),j(box_r),k(-box_r):k(box_r)] = 1
mask[i(-box_r):i(box_r),j(-box_r),k(-box_r):k(box_r)] = 1
mask[i(-box_r):i(box_r),j(-box_r):j(box_r),k(box_r)] = 1
mask[i(-box_r):i(box_r),j(-box_r):j(box_r),k(-box_r)] = 1

phi0_mat[mask == 0] = 0

mask[ground_cor] = 1
phi0_mat[ground_cor] = 0

phi0_args = (phi0_mat, mask)

(q3_phi_sol, n, sec_elapsed) = relaxed_poisson(phi0_args, rho0, h, 1e-4)

print(f"phi complete; n: {n}, total time: {sec_elapsed} sec.")

start = time()
q3_f = e_field(q3_phi_sol)
stop = time()

print(f"E complete; total time: {stop-start} sec.")

start = time()
q3_rho_sol = rho(q3_phi_sol)
stop = time()

print(f"rho complete; total time: {stop-start} sec.")

plt.figure(dpi=150)
plt.imshow(q3_phi_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\Phi$")
plt.savefig("q3_phi_inside.png")
plt.show()

plt.figure(dpi=150)
plt.quiver(Y[i(0),:,:], Z[i(0),:,:], q3_f[i(0),:,:,1], q3_f[i(0),:,:,2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("q3_E_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q3_rho_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q3_rho_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q3_rho_sol[i(-box_r),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q3_rho_surface.png")
plt.show()

point_charge_pot = 1/np.sqrt(X**2 + Y**2 + Z**2)
f_point = e_field(point_charge_pot)
masked_f = np.array(f_point)
masked_f[X**2+Y**2+Z**2 < 0.2] = 0

plt.figure(dpi=150)
plt.quiver(Y[i(0),:,:], Z[i(0),:,:], masked_f[i(0),:,:,1], masked_f[i(0),:,:,2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("q3_only_charge_E_inside.png")
plt.show()

# ====================== with point charge inside ======================

rho0[i(0),j(0),k(0)] = 1/h**3

(q3_charge_phi_sol, n, sec_elapsed) = relaxed_poisson(phi0_args, rho0, h, 1e-4)

print(f"phi complete; n: {n}, total time: {sec_elapsed} sec.")

start = time()
q3_charge_f = e_field(q3_charge_phi_sol)
stop = time()

print(f"E complete; total time: {stop-start} sec.")

start = time()
q3_charge_rho_sol = rho(q3_charge_phi_sol)
stop = time()

print(f"rho complete; total time: {stop-start} sec.")

plt.figure(dpi=150)
plt.imshow(q3_charge_phi_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\Phi$")
plt.savefig("q3_charge_phi_inside.png")
plt.show()

masked_f = np.array(q3_charge_f)
masked_f[X**2+Y**2+Z**2 < 0.2] = 0
plt.figure(dpi=150)
plt.quiver(Y[i(0),:,:], Z[i(0),:,:], masked_f[i(0),:,:,1], masked_f[i(0),:,:,2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("q3_charge_E_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q3_charge_rho_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q3_charge_rho_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q3_charge_rho_sol[i(-box_r),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q3_charge_rho_surface.png")
plt.show()

# ============================ only point charge ============================

rho0[i(0),j(0),k(0)] = 1/h**3

phi0_mat = np.zeros(X.shape)
mask = np.zeros(phi0_mat.shape)
mask[ground_cor] = 1
phi0_mat[ground_cor] = 0

phi0_args = (phi0_mat, mask)

(charge_nopot_phi_sol, n, sec_elapsed) = relaxed_poisson(phi0_args, rho0, h, 1e-3)

print(f"phi complete; n: {n}, total time: {sec_elapsed} sec.")

start = time()
charge_nopot_f = e_field(charge_nopot_phi_sol)
stop = time()

print(f"E complete; total time: {stop-start} sec.")

start = time()
charge_nopot_rho_sol = rho(charge_nopot_phi_sol)
stop = time()

print(f"rho complete; total time: {stop-start} sec.")

plt.figure(dpi=150)
plt.imshow(charge_nopot_phi_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\Phi$")
plt.savefig("q3_charge_nopot_phi_inside.png")
plt.show()

masked_f = np.array(charge_nopot_f)
masked_f[X**2+Y**2+Z**2 < 0.3] = 0
plt.figure(dpi=150)
plt.quiver(Y[i(0),:,:], Z[i(0),:,:], masked_f[i(0),:,:,1], masked_f[i(0),:,:,2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("q3_charge_nopot_E_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(charge_nopot_rho_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q3_charge_nopot_rho_inside.png")
plt.show()

plt.figure(dpi=150)
plt.plot(x, q3_charge_phi_sol[:,j(0),k(0)], label='charge in box')
plt.plot(x, q3_phi_sol[:,j(0),k(0)], label='only box')
plt.plot(x, point_charge_pot[:,j(0),k(0)], '--', label=r'$\phi=\frac{1}{r}$')
# plt.plot(x, charge_nopot_phi_sol[:,j(0),k(0)], label='only charge')
plt.grid()
plt.legend()
plt.xlabel(r'x')
plt.ylabel(r'$\Phi$')
plt.savefig("q3_potentials.png")
plt.show()

#%% q4


rho0 = np.zeros(X.shape)

phi0_mat = np.zeros(X.shape)

phi_s = 1
phi0_mat[i(box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r),j(-box_r):j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r),k(-box_r):k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r):j(box_r),k(box_r)] = phi_s
phi0_mat[i(-box_r):i(box_r),j(-box_r):j(box_r),k(-box_r)] = phi_s

phi0_mat[i(box_r),j(-box_r/2):j(box_r/2),k(-box_r/2):k(box_r/2)] = 0
phi0_mat[i(-box_r),j(-box_r/2):j(box_r/2),k(-box_r/2):k(box_r/2)] = 0
phi0_mat[i(-box_r/2):i(box_r/2),j(box_r),k(-box_r/2):k(box_r/2)] = 0
phi0_mat[i(-box_r/2):i(box_r/2),j(-box_r),k(-box_r/2):k(box_r/2)] = 0
phi0_mat[i(-box_r/2):i(box_r/2),j(-box_r/2):j(box_r/2),k(box_r)] = 0
phi0_mat[i(-box_r/2):i(box_r/2),j(-box_r/2):j(box_r/2),k(-box_r)] = 0

phi0_mat[ground_cor] = 0

mask = np.zeros(phi0_mat.shape)
mask[phi0_mat != 0] = 1
mask[ground_cor] = 1
phi0_args = (phi0_mat, mask)

(q4_phi_sol, n, sec_elapsed) = relaxed_poisson(phi0_args, rho0, h, (1/5)*1e-2)

print(f"phi complete; n: {n}, total time: {sec_elapsed} sec.")

start = time()
q4_f = e_field(q4_phi_sol)
stop = time()

print(f"E complete; total time: {stop-start} sec.")

start = time()
q4_rho_sol = rho(q4_phi_sol)
stop = time()

print(f"rho complete; total time: {stop-start} sec.")

plt.figure(dpi=150)
plt.imshow(q4_phi_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\Phi$")
plt.savefig("q4_phi_inside.png")
plt.show()

plt.figure(dpi=150)
plt.quiver(Y[i(0),:,:], Z[i(0),:,:], q4_f[i(0),:,:,1], q4_f[i(0),:,:,2])
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.xlabel("y")
plt.ylabel("z")
plt.savefig("q4_E_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q4_rho_sol[i(0),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q4_rho_inside.png")
plt.show()

plt.figure(dpi=150)
plt.imshow(q4_rho_sol[i(-box_r),:,:])
plt.xlabel("y")
plt.ylabel("z")
plt.colorbar(label=r"$\rho$")
plt.savefig("q4_rho_surface.png")
plt.show()
