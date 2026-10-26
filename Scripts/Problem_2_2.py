import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, LinearOperator
import sys
from numba import njit, prange
import utils
from dataclasses import dataclass, field
from matplotlib.patches import Circle
import os

@dataclass
class ImBound:
    nodes: np.typing.NDArray[np.float64]
    sigma: float
    num_nodes: int = field(init=False)
    forces: np.typing.NDArray[np.float64] = field(init=False)
    #forces: list[np.typing.NDArray[np.float64]] = field(init=False)

    def __post_init__(self):
        self.num_nodes = self.nodes.shape[1]

    def evaluate(self,X,Y):
        self.forces = np.zeros((X.shape[0], X.shape[1], self.num_nodes))
        x = np.stack([X, Y], axis=0) 
        for i in range(self.num_nodes):
            force = G(x, self.nodes[:,i], self.sigma)
            self.forces[:,:,i] = force

    # def evaluate(self,X,Y):
    #         self.forces = [] #np.zeros((X.shape[0], X.shape[1], self.num_nodes))
    #         x = np.stack([X, Y], axis=-1) 
    #         for i in range(self.num_nodes):
    #             force = G(x, self.nodes[:,i], self.sigma)
    #             self.forces.append(force)

@njit(fastmath=True, cache=True)
def G(x, y, sigma):
    '''
    x is collocation point
    y is source point
    '''
    return 1/(2.0*np.pi*sigma**2)*np.exp(-((x[0]-y[0])**2+(x[1]-y[1])**2)/(2.0*sigma**2))

@njit(parallel=True,fastmath=True, cache=True)
def applyA(v, N_FD, dx,nodes,sigma,num_nodes,Gs):
    u, q = v[:N_FD**2],v[N_FD**2:]
    u = u.reshape((N_FD, N_FD))
    phi = np.zeros_like(u) # Update to FD nodes
    psi = np.zeros_like(q) # Update to charge weights
    
    # Apply Lu - Sq
    for i in prange(N_FD):
        for j in range(N_FD):
            u_c = u[i, j]
            u_d = u[i-1, j] if i > 0   else 0.0   # homogeneous Dirichlet
            u_u = u[i+1, j] if i < N_FD-1 else 0.0
            u_l = u[i, j-1] if j > 0   else 0.0
            u_r = u[i, j+1] if j < N_FD-1 else 0.0
            phi[i, j] = 1/(dx**2) * (u_u + u_d + u_l + u_r - 4.0 * u_c) - np.dot(q, Gs[i,j,:])
    
    # Apply Ju
    for k in prange(num_nodes):
        for i in range(N_FD):
            for j in range(N_FD):
                psi[k] += u[i,j]*Gs[i,j,k] # Check that this is actually correct

    b = np.concatenate((phi.ravel(), psi.ravel()))

    return b

def imBoundSolver(N_FD, bounds, center, f, g, verbose=False, N_ib = None, tol =1e-5):
    # Build RHS
    # Laplacian RHS
    x = np.linspace(bounds[0], bounds[1], N_FD+2)
    x = x[1:-1] # truncate x to not include BCs
    dx = x[1]-x[0]
    if N_ib == None:
        N_ib = np.floor(2.0*np.pi/dx).astype(int)
    X,Y = np.meshgrid(x,x, indexing="xy")
    b1 = f(X,Y)

    # Boundary Conditions
    b1[0, :] -= 0 / dx**2 # Left hand BC
    b1[-1, :] -= 0 / dx**2 # right hand BC
    b1[:, 0] -= np.sin(2.0*Y[:,0]) / dx**2 # top BC
    b1[:, -1] -= np.sin(2.0*Y[:,-1]) / dx**2# right BC

    # Interpolation RHS
    ib = ImBound(nodes = utils.getCircle(N_ib,center),sigma=1.2*dx)
    ib.evaluate(X,Y)
    b2 = g(ib.nodes[0,:],ib.nodes[1,:])/dx**2

    b = np.concatenate([b1.ravel(), b2.ravel()])

    # Build the Linear system
    mat_vec = LinearOperator(((N_FD)**2+N_ib,(N_FD)**2+N_ib), matvec=lambda v: applyA(v, N_FD, dx, ib.nodes, ib.sigma, ib.num_nodes, ib.forces), dtype=np.float64)

    it = utils.IterCounter(verbose)

    #Sovle
    sol,exit_code = gmres(mat_vec, b.flatten(), rtol=tol, restart=200, maxiter=10000, callback=it)
    u_approx = processSolution(sol, N_FD,)
    result = {"solution": u_approx,"xgrid": X, "ygrid": Y, "ImBound": ib}
    return result

def processSolution(sol, N_FD):
    u, q = sol[:N_FD**2], sol[N_FD**2:]
    u = u.reshape((N_FD,N_FD))
    return u

def plot_results(X, Y, u_approx, u_exact, ib, center, colormap="viridis", outdir="Images"):
# Written by claude

    os.makedirs(outdir, exist_ok=True)
    radius = np.mean(np.hypot(ib.nodes[0] - center[0], ib.nodes[1] - center[1]))

    # --- 1. Numerical vs exact, side by side ---
    vmin = min(u_approx.min(), u_exact.min())
    vmax = max(u_approx.max(), u_exact.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, data, title in zip(axes, [u_approx, u_exact], ["Numerical solution", "Exact solution"]):
        pc = ax.pcolormesh(X, Y, data, cmap=colormap, shading="auto", vmin=vmin, vmax=vmax)
        ax.add_patch(Circle(center, radius, fill=False, color="k", lw=1))
        ax.plot(ib.nodes[0], ib.nodes[1], "kx", ms=5)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y")
    fig.colorbar(pc, ax=axes, shrink=0.85)
    fig.savefig(os.path.join(outdir, "solution_comparison.png"), dpi=200)

    # --- 2. Absolute error ---
    err = np.abs(u_approx - u_exact)
    fig, ax = plt.subplots(figsize=(6, 5))
    pc = ax.pcolormesh(X, Y, err, cmap=colormap, shading="auto")
    ax.add_patch(Circle(center, radius, fill=False, color="w", lw=1))
    ax.set_title(f"Absolute error (max = {err.max():.2e})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax)
    fig.savefig(os.path.join(outdir, "error.png"), dpi=200)

    # --- 3. 3D surface of the numerical solution ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, u_approx, cmap=colormap)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
    ax.set_title("Numerical solution")
    fig.colorbar(surf, shrink=0.6, pad=0.1)
    fig.savefig(os.path.join(outdir, "surface.png"), dpi=200)

    plt.show()

def main():
    # parameters
    N_FD = 50
    #N_ib = 10
    bounds = [0,2.0*np.pi]
    center = [np.pi, np.pi]
    f = lambda x,y : -8.0 * np.sin(2.0*y) * np.cos(2.0*x)
    g = lambda x,y : np.cos(2.0*x)*np.sin(2.0*y)
    solution = lambda x,y : np.cos(2.0*x)*np.sin(2.0*y)
    verbosity = True
    colormap = "inferno"

    sol = imBoundSolver(N_FD, bounds, center, f, g, verbose=True, tol=1e-5)

    u_approx = sol["solution"]
    X, Y = sol["xgrid"], sol["ygrid"]

    # Plot solution and Error
    u_exact = solution(X,Y)

    plot_results(X, Y, u_approx, u_exact, sol["ImBound"], center, colormap)

    # Get Error Plot 
    O_IM = lambda N_FD: imBoundSolver(N_FD, bounds, center, f, g, tol = 1e-5)
    utils.convTest([10, 300], 8, solution, O_IM)

    return 0

if __name__ == "__main__":
    main()