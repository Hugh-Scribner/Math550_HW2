import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, LinearOperator
import sys
from numba import njit, prange

class IterCounter:
    """GMRES callback: counts iterations (SciPy doesn't return ITER like MATLAB)
    and, if verbose, prints the iteration number and relative residual ||r||/||f||.
    `step` is the number of iterations per callback: 1 for SciPy, `restart` for
    CuPy (which calls back once per restart cycle)."""
    def __init__(self, verbose=False, step=1):
        self.n = 0
        self.verbose = verbose
        self.step = step
    def __call__(self, rel_res):
        self.n += self.step
        if self.verbose:
            print(f"    iteration {self.n:5d}: relative residual {float(rel_res):.3e}")

#@njit(parallel=True)
def applyLap(u, N, dx):
    u = u.reshape((N, N))
    y = np.zeros_like(u)
    for i in prange(N):
        for j in range(N):
            u_c = u[i, j]
            u_d = u[i-1, j] if i > 0   else 0.0   # homogeneous Dirichlet
            u_u = u[i+1, j] if i < N-1 else 0.0
            u_l = u[i, j-1] if j > 0   else 0.0
            u_r = u[i, j+1] if j < N-1 else 0.0
            y[i, j] = 1/(dx*dx) * (u_u + u_d + u_l + u_r - 4.0 * u_c)
    return y.ravel()

def lapSolver(N, bounds, f, verbose):
    x = np.linspace(bounds[0], bounds[1], N+2)
    dx = x[1]-x[0]
    x = x[1:-1] # truncate x to not include BCs
    
    # Build the Linear system
    mat_vec = LinearOperator((N**2,N**2), matvec=lambda v: applyLap(v, N, dx))
        
    # Build the RHS
    X,Y = np.meshgrid(x,x, indexing="ij")
    b = f(X,Y)
    b[0, :] -= np.sin(2.0*Y[0,:]) / dx**2 # Left hand BC
    b[-1, :] -= np.sin(2.0*Y[-1,:]) / dx**2 # right hand BC
    b[:, 0] -= 0 / dx**2 # top BC
    b[:, -1] -= 0 / dx**2# right BC
    
    it = IterCounter(verbose)
    u_approx,exit_code = gmres(mat_vec, b.flatten(), callback=it)
    u_approx = u_approx.reshape(N,N)
    
    result = {
        "xgrid": X,
        "ygrid": Y,
        "solution": u_approx
    }
    
    return result

def rel_error(u_exact, u_approx):
    L_2_error = np.linalg.norm(u_exact-u_approx, ord = 'fro')/np.linalg.norm(u_exact, ord = 'fro')
    return L_2_error

def convTest(mesh_range, num_trials, exact_sol, num_Operator, filepath = 'Images\\convergence_plot.png'):
    meshes = np.floor(np.linspace(mesh_range[0], mesh_range[1], num_trials)).astype(int)
    L_2 = np.zeros(num_trials,)
    for i in range(num_trials):
        print(f"Simulating mesh with {meshes[i]**2} nodes")
        result = num_Operator(meshes[i])
        u_approx = result["solution"]
        xx, yy = result["xgrid"], result["ygrid"]
        u_exact = exact_sol(xx,yy)
        
        L_2[i] = rel_error(u_exact, u_approx) #calculate relative errors for increasingly finer meshes.
    fig3 = plt.figure(3)
    ax3 = plt.axes()
    plt.rcParams['lines.linewidth'] = 3 
    plt.loglog(meshes, 1/meshes, label = r"$\mathcal{O}(\Delta x)$ Reference", color = 'black', linestyle = '--')
    plt.loglog(meshes, 1/meshes**2, label = r"$\mathcal{O}(\Delta x^2)$ Reference", color = 'black')
    plt.loglog(meshes, L_2, label = "Rel Error", color = 'red')
    plt.legend()
    ax3.set_xlabel("Number of FD Nodes")
    ax3.set_ylabel("Relative Error")
    plt.savefig(filepath)
    return 0


def main():

    # parameters
    N = 40
    bounds = [0,np.pi]
    f = lambda x,y : -8.0 * np.sin(2.0*y) * np.cos(2.0*x)
    solution = lambda x,y : np.cos(2.0*x)*np.sin(2.0*y)
    verbose = True
    colormap = "viridis"

    # Get Numerical Solution
    sol = lapSolver(N, bounds, f, verbose)
    X,Y,u_approx = sol["xgrid"],sol["ygrid"],sol["solution"]

    # Get Exact Solution
    u_exact = solution(X,Y)

    # Plot both solutions to compare visually
    fig1, ax1 = plt.subplots(nrows=1, ncols=2, constrained_layout=True)
    vmin = min(u_exact.min(), u_approx.min())
    vmax = max(u_exact.max(), u_approx.max())
    im0 = ax1[0].pcolormesh(X, Y, u_exact, cmap=colormap, vmin=vmin, vmax=vmax)
    im1 = ax1[1].pcolormesh(X, Y, u_approx, cmap=colormap, vmin=vmin, vmax=vmax)
    ax1[0].set_title('Exact Solution')
    ax1[1].set_title('Approx. Solution')
    for ax in ax1:
        ax.set_aspect('equal')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.xaxis.set_major_locator(MultipleLocator(1))
    fig1.colorbar(im0, ax=ax1, orientation='vertical', fraction=0.046, pad=0.04,  shrink=0.6)
    plt.savefig("Images\\problem4_1.png", bbox_inches='tight')

    # Get Error Plot
    O_lap = lambda N: lapSolver(N, bounds, f, False)
    convTest([10, 200], 5, solution, O_lap)

    return 0

if __name__ == "__main__":
    main()