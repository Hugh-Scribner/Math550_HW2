import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import scipy.sparse as sp
from scipy.sparse.linalg import gmres, LinearOperator
from scipy.interpolate import RegularGridInterpolator
import sys
from numba import njit, prange
import utils
from dataclasses import dataclass, field
from matplotlib.patches import Circle
import glob
import imageio.v2 as imageio
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

def imBoundSolver(N_FD, bounds, center, f, g, verbose=False, N_ib = None, tol =1e-5, r = 1):
    # Build RHS
    # Laplacian RHS
    x = np.linspace(bounds[0], bounds[1], N_FD+2)
    x = x[1:-1] # truncate x to not include BCs
    dx = x[1]-x[0]
    if N_ib is None:
        N_ib = np.floor(2.0*np.pi*r/dx).astype(int)
        if verbose:
            print(f"N_ib is {N_ib}")
    X,Y = np.meshgrid(x,x, indexing="xy")
    b1 = f(X,Y)

    # Boundary Conditions
    b1[0, :] -= 0 / dx**2 # Left hand BC
    b1[-1, :] -= 0 / dx**2 # right hand BC
    b1[:, 0] -= 0 / dx**2 # top BC
    b1[:, -1] -= 0 / dx**2# right BC

    # Interpolation RHS
    ib = ImBound(nodes = utils.getCircle(N_ib,center, r),sigma=1.2*dx)
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

def plot_results(X, Y, u_approx, nodes, center, colormap="viridis",
                 vlim=None, save_path=None, show=False):
    radius = np.mean(np.hypot(nodes[0] - center[0], nodes[1] - center[1]))

    interp = RegularGridInterpolator((Y[:, 0], X[0, :]), u_approx,
                                     bounds_error=False, fill_value=None)

    t = np.linspace(0, 2*np.pi, 400)
    cx = center[0] + radius*np.cos(t)
    cy = center[1] + radius*np.sin(t)
    cz = interp(np.column_stack([cy, cx]))
    nz = interp(np.column_stack([nodes[1], nodes[0]]))

    vmin, vmax = vlim if vlim is not None else (u_approx.min(), u_approx.max())
    offset = 0.01 * (vmax - vmin)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(X, Y, u_approx, cmap=colormap, alpha=0.9,
                           vmin=vmin, vmax=vmax)
    ax.plot(cx, cy, cz + offset, "k-", lw=2, label="Immersed boundary")
    ax.scatter(nodes[0], nodes[1], nz + offset, c="r", s=10, depthshade=False,
               label="Boundary nodes")
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("u")
    ax.set_title("Numerical solution")
    ax.legend(loc="upper left")
    fig.colorbar(surf, shrink=0.6, pad=0.1)

    if save_path is not None:
        fig.savefig(save_path, dpi=200)   # 1600x1200, divisible by 16 (good for mp4)
    if show:
        plt.show()
    plt.close(fig)                        # important: avoids memory buildup over 180 frames


def make_movie(frame_dir, out_path, fps):
    files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    with imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8) as writer:
        for fn in files:
            writer.append_data(imageio.imread(fn))


def main():
    N_FD = 50
    fps = 60
    seconds = 9
    loop_time = 3                     # seconds per revolution of the circle center

    bounds = [-np.pi, np.pi]
    frames = fps * seconds
    # endpoint=False so the last frame doesn't duplicate the first (seamless loop)
    theta = np.linspace(0, seconds/loop_time * 2.0*np.pi, frames, endpoint=False)
    center = [0,0]
    f = lambda x, y: x*0
    g = lambda x, y: np.cos(2.0*x)*np.sin(2.0*y)
    colormap = "inferno"

    frame_dir = os.path.join("Images", "movie_frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Pass 1: solve every frame (store only u and the node coordinates, not the big Gs array)
    results = []
    for i in range(frames):
        radius = .5*np.sin(theta[i]) + 1
        sol = imBoundSolver(N_FD, bounds, center, f, g, verbose=False, tol=1e-5, r = radius)
        results.append((sol["solution"], sol["ImBound"].nodes.copy(), center))
        print(f"solved {i+1}/{frames}")
    X, Y = sol["xgrid"], sol["ygrid"]

    # Global limits so color and z-axis are constant across frames
    vlim = (min(r[0].min() for r in results), max(r[0].max() for r in results))

    # Pass 2: render frames
    for i, (u, nodes, center) in enumerate(results):
        path = os.path.join(frame_dir, f"frame_{i:04d}.png")
        plot_results(X, Y, u, nodes, center, colormap, vlim=vlim, save_path=path)
        print(f"rendered {i+1}/{frames}")

    make_movie(frame_dir, os.path.join("Images", "Radium_Movie.mp4"), fps)
    return 0

if __name__ == "__main__":
    main()