import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, LinearOperator
import sys
from numba import njit, prange
import utils

def G(x, y, sigma):
    '''
    x is collocation point
    y is source point
    '''
    return 1/(2.0*np.pi*sigma**2)*np.exp(-((x[0]-y[0]**2+(x[1]-y[1])**2)/(2.0*sigma**2)))

#@njit(parallel=True)
def applyA(v, ib_nodes, N_FD, N_IB, sigma, dx):
    u, q = v[:N_FD],v[N_FD:]
    u = u.reshape((N_FD, N_FD))
    phi = np.zeros_like(u) # Update to FD nodes
    psi = np.zeros_like(q) # Update to charge weights
    for i in prange(N_FD):
        for j in range(N_FD):
            u_c = u[i, j]
            u_d = u[i-1, j] if i > 0   else 0.0   # homogeneous Dirichlet
            u_u = u[i+1, j] if i < N-1 else 0.0
            u_l = u[i, j-1] if j > 0   else 0.0
            u_r = u[i, j+1] if j < N-1 else 0.0
            x = np.array([(j+1)*dx, (i+1)*dx])
            phi[i, j] = 1/(dx**2) * (u_u + u_d + u_l + u_r - 4.0 * u_c) - np.dot(q, G(x, ib_nodes, sigma))
    for i in range(N_IB):
        psi[i] = 1
    return y.ravel()

def lapSolver(N, bounds, f, verbose):
    x = np.linspace(bounds[0], bounds[1], N+2)
    dx = x[1]-x[0]
    x = x[1:-1] # truncate x to not include BCs
    
    # Build the Linear system
    mat_vec = LinearOperator((N**2,N**2), matvec=lambda v: applyA(v, N, dx))
        
    # Build the RHS
    X,Y = np.meshgrid(x,x, indexing="ij")
    b = f(X,Y)
    b[0, :] -= np.sin(2.0*Y[0,:]) / dx**2 # Left hand BC
    b[-1, :] -= np.sin(2.0*Y[-1,:]) / dx**2 # right hand BC
    b[:, 0] -= 0 / dx**2 # top BC
    b[:, -1] -= 0 / dx**2# right BC
    
    it = utils.IterCounter(verbose)
    u_approx,exit_code = gmres(mat_vec, b.flatten(), callback=it)
    u_approx = u_approx.reshape(N,N)
    result = {"xgrid": X,"ygrid": Y,"solution": u_approx}
    return result

def G(x, y, sigma):
    '''
    x is collocation point
    y is source point
    '''
    return 1/(2.0*np.pi*sigma**2)*np.exp(-((x[0]-y[0])**2+(x[1]-y[1])**2)/(2.0*sigma**2))

def main():
    # parameters
    N = 50
    bounds = [0,2.0*np.pi]
    center = [np.pi, np.pi]
    f = lambda x,y : -8.0 * np.sin(2.0*y) * np.cos(2.0*x)
    solution = lambda x,y : np.cos(2.0*x)*np.sin(2.0*y)
    verbose = True
    colormap = "viridis"

    # Get Numerical Solution
    cir = utils.getCircle(10,center)
    x = np.linspace(bounds[0], bounds[1], N)
    X,Y = np.meshgrid(x,x, indexing = 'xy')
    mesh=[X,Y]
    Z = np.sum(G([X[..., None], Y[..., None]], cir, .2), axis=-1)
    #Z = np.sum(G(mesh, cir, 1))
    plt.figure()
    plt.pcolormesh(X, Y, Z, cmap=colormap)
    plt.savefig("Images\\gaussian_boundary.png")

    return 0

if __name__ == "__main__":
    main()